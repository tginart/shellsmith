"""Backprop/W&B logging, independent of optimizer/checkpoint publication.

The trainer owns one selected sink. SDKs journal before upload; Shellsmith
keeps its recovery files and replays missed metrics/finished episodes on resume.
Tables contain text trajectories, never weights or token-exact replay arrays.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib
from importlib.machinery import PathFinder
import json
import math
import os
import queue
from pathlib import Path
import shlex
import sys
import threading
import time
from typing import Any

from shellsmith.episode.progress import add_listener, remove_listener
from shellsmith.process_io import write_json
from shellsmith.paths import PROJECT_ROOT
from shellsmith.train.config import TrainConfig
from shellsmith.train.durable_capture import load_durable_outcome
from shellsmith.train.observability import RunObservations, SdkHealth
from shellsmith.train.observation_panels import build_panels

LOCAL_CONNECTION_FILE = PROJECT_ROOT / ".backprop" / "config.json"
MAX_TRAJECTORY_BYTES = 131072


class _OfficialWandbImports:
    """Resolve installed W&B modules before Backprop's compatibility guards.

    Backprop 0.1.0 installs guards for unsupported ``wandb.*`` aliases even
    when imported natively. Those aliases must not block the real W&B SDK.
    This temporary finder uses Python's normal filesystem importer and leaves
    all Backprop modules and their guards intact.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wandb" or fullname.startswith("wandb."):
            return PathFinder.find_spec(fullname, path, target)
        return None


def _load_sdk(provider: str):
    if provider != "wandb":
        return importlib.import_module(provider)
    finder = _OfficialWandbImports()
    sys.meta_path.insert(0, finder)
    try:
        sdk = importlib.import_module(provider)
    finally:
        sys.meta_path.remove(finder)
    if sdk.__name__ != "wandb":
        raise RuntimeError("The wandb provider requires the official W&B SDK; start training without a Backprop compatibility launcher.")
    return sdk


def _connection_file(config: TrainConfig) -> str | None:
    if config.telemetry.provider != "backprop":
        return None
    explicit = os.environ.get("BACKPROP_CONFIG") or config.telemetry.connection_file
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    return str(LOCAL_CONNECTION_FILE) if LOCAL_CONNECTION_FILE is not None and LOCAL_CONNECTION_FILE.is_file() else None


@contextmanager
def _sdk_connection(path: str | None):
    """Use the public SDK environment contract only during initialization.

    Backprop captures its deployment at init. Restore the parent environment
    before any rollout/GM process starts, including on authentication failure.
    """
    previous = os.environ.get("BACKPROP_CONFIG")
    if path is not None:
        os.environ["BACKPROP_CONFIG"] = path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("BACKPROP_CONFIG", None)
        else:
            os.environ["BACKPROP_CONFIG"] = previous


_OMIT = {
    "prompt_token_ids", "sampled_tokens", "sampled_logprobs", "behavior_logprobs",
    "advantages", "loss_mask", "logprobs", "token_ids", "input_ids",
    "api_key", "access_token", "authorization", "weights", "source_spans",
    "logprob_target_indices", "logprobs", "password", "secret", "token",
    "backprop_token", "foundry_token", "tinker_api_key", "wandb_api_key",
}


def _compact(value: Any) -> Any:
    """Retain text/tools/assessments while omitting bulky replay and credentials."""
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key.lower() not in _OMIT and not key.lower().endswith(("_api_key", "_password", "_secret", "_token"))}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _json(value: Any) -> str:
    return json.dumps(_compact(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _scalars(value: dict[str, Any], prefix: str) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(_scalars(item, name))
        elif isinstance(item, (int, float)) and math.isfinite(item):
            result[name] = item
    return result


def is_telemetry_enabled(config: TrainConfig) -> bool:
    """
    Purpose: Apply explicit and provider-environment disable modes before I/O.
    Input: The telemetry configuration and the provider environment.
    Output: True when the selected provider is enabled.
    """
    mode = config.telemetry.mode
    if mode == "auto" and config.telemetry.provider == "wandb":
        mode = os.environ.get("WANDB_MODE", mode)
    return mode != "disabled"


def require_telemetry(config: TrainConfig) -> None:
    """Check the lazy dependency before starting paid work or writing a run."""
    if is_telemetry_enabled(config):
        provider = config.telemetry.provider
        try:
            _load_sdk(provider)
        except ImportError as error:
            install = "pip install -e '.[wandb]'" if provider == "wandb" else "pip install -e ."
            raise RuntimeError(
                f"Training telemetry provider {provider!r} requires its SDK. Install with "
                f"{install}, or set [telemetry].mode = 'disabled'."
            ) from error


def persist_publication(root: Path, row: dict, *, enabled: bool) -> bool:
    """
    Purpose: Save a publication receipt without a failure in the trainer transaction.
    Input: The run root, the receipt, and the telemetry enable state.
    Output: True if the receipt is durable. False if telemetry is disabled or I/O fails.
    """
    if not enabled:
        return False
    identity = hashlib.sha256(_json(row).encode()).hexdigest()
    prefix = "checkpoint" if row.get("kind") == "state_checkpoint" else "policy"
    try:
        write_json(root / "policy-publications" / f"{prefix}-{identity}.json", row)
    except OSError as error:
        print(f"shellsmith: publication telemetry receipt failed: {error}",
              file=sys.stderr, flush=True)
        return False
    return True


class TrainingTelemetry:
    """One selected experiment run, with replayable local publication receipts.

    Logging failures warn and leave the durable source for retry. They cannot
    change whether an optimizer transaction or rollout was accepted. Receipts
    are recorded only after SDK journaling, so crash recovery is at least once.
    """

    def __init__(self, config: TrainConfig, *, snapshot: dict, state: dict,
                 segment: int | None = None):
        self.config, self.snapshot, self.state = config, snapshot, state
        self.root = config.resolved_log_dir()
        self.segment = len(state["segments"]) - 1 if segment is None else segment
        self.run = self.sdk = None
        self.lock = threading.RLock()
        self.seen: set[str] = set()
        self.errors = 0
        self.started = time.monotonic()
        self.restored_offset = state["next_world_offset"] + len(state.get("settled_world_offsets", []))
        self.mode = config.telemetry.mode
        self.provider = config.telemetry.provider
        self.label = "Backprop" if self.provider == "backprop" else "W&B"
        self.connection_file = _connection_file(config)
        self.tables: dict[str, Any] = {}
        self.table_rows: dict[str, set[str]] = {}
        self.pending_receipts: set[str] = set()
        self.queue = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.worker = None
        self.dropped = 0
        self.last_progress = {}
        self.next_progress = 0.0
        self.counter_lock = threading.Lock()
        self.gm_calls = 0
        self.gm_seconds = 0.0
        self.observations = None
        self.collector = None
        self.collector_errors = 0
        self.next_figures = 0.0


    def __enter__(self):
        options = self.config.telemetry
        if not is_telemetry_enabled(self.config):
            self.mode = "disabled"
            return self
        self.sdk = _load_sdk(self.provider)
        if self.mode == "auto":
            if self.provider == "wandb":
                self.mode = os.environ.get("WANDB_MODE") or "online"
                if self.mode == "dryrun":
                    self.mode = "offline"
            else:
                self.mode = "online" if (
                    self.connection_file or (
                        os.environ.get("BACKPROP_HOST") and os.environ.get("BACKPROP_STREAM_DATASET_RID")
                    )
                ) else "offline"
        if self.mode == "disabled":
            return self
        with _sdk_connection(self.connection_file):
            self.run = self._init_run(options)
        try:
            self._start()
        except BaseException:
            remove_listener(self.progress)
            try:
                self.run.finish(exit_code=1)
            except Exception as error:
                self._warn(error)
            raise
        return self

    def _init_run(self, options):
        return self.sdk.init(
            project=options.project, entity=options.entity or None,
            id=self.config.run.run_id, name=self.config.run.run_id,
            group=self.config.run.run_id, job_type="train",
            tags=["shellsmith", self.config.compute.backend],
            dir=str(self.root / self.provider), mode=self.mode,
            resume="allow" if self.mode == "online" else None,
            config=_compact({"shellsmith": self.config.to_jsonable(), "provenance": self.snapshot,
                             "observability": {"version": 1}}),
            settings=self.sdk.Settings(
                disable_git=True, save_code=False, disable_code=True,
                console="wrap" if options.capture_console else "off",
                x_disable_stats=not options.system_metrics,
                x_stats_sampling_interval=options.system_interval_seconds,
                init_timeout=options.init_timeout_seconds, finish_timeout=15.0,
                reinit="create_new",
            ),
        )

    def _start(self):
        options = self.config.telemetry
        deployment = json.loads(Path(self.connection_file).read_text()) if self.connection_file else {}
        # Separate receipts for offline/online destinations. Moving online
        # replays local sources, including data from earlier offline runs.
        destination = ([
            {key: deployment.get(key) for key in ("host", "stream_dataset_rid", "media_dataset_rid", "branch")},
            os.environ.get("BACKPROP_HOST"), os.environ.get("BACKPROP_STREAM_DATASET_RID"),
        ] if self.provider == "backprop" else [os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")])
        namespace = hashlib.sha256(_json([
            options.project, self.run.entity, self.mode, *destination,
        ]).encode()).hexdigest()[:16]
        self.receipts = self.root / self.provider / f"published-{namespace}.jsonl"
        if self.receipts.exists():
            self.seen = {line for line in self.receipts.read_text().splitlines() if line}
        metadata_path = self.root / (self.provider + ".json")
        prior = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        self.sync_dir = self.run.sync_dir if self.provider == "backprop" else self.run.settings.sync_dir
        self.url = self.run.url if self.mode == "online" else None
        journals = list(dict.fromkeys([*prior.get("sync_dirs", []), self.sync_dir]))
        write_json(metadata_path, {
            "provider": self.provider, "project": options.project,
            "entity": self.run.entity, "run_id": self.run.id, "mode": self.mode,
            "url": self.url, "sync_dir": self.sync_dir, "sync_dirs": journals,
            "connection_file": self.connection_file,
        })
        self.observations = RunObservations(self.config, self.state, self.segment)
        self.sdk_health = SdkHealth(Path(self.sync_dir))
        self.run.define_metric("run/elapsed_seconds", hidden=True)
        self.run.define_metric("train/step", hidden=True)
        self.run.define_metric("train/*", step_metric="train/optimizer_updates_committed")
        for prefix in ("inference", "inference_service", "gm", "rollout", "telemetry", "compute", "run"):
            self.run.define_metric(prefix + "/*", step_metric="run/elapsed_seconds")
        self.run.define_metric("train/optimizer_updates_committed", step_metric="run/elapsed_seconds")
        for prefix in ("forward", "optim", "mlx", "execution",
                       "backward_execution", "optimizer_execution"):
            self.run.define_metric(prefix + "/*", step_metric="train/optimizer_updates_committed")
        self.run.log({**self.observations.context(), "telemetry/segment": self.segment, "train/step": self.state["step"],
                      "telemetry/restored_world_offset": self.state["next_world_offset"]})
        add_listener(self.progress)
        print(f"shellsmith: {self.label} {self.mode}: {self.url or self.sync_dir}", flush=True)
        if self.mode == "offline":
            command = f"python -m {self.provider} sync {shlex.quote(self.sync_dir)}"
            if self.connection_file:
                command = f"BACKPROP_CONFIG={shlex.quote(self.connection_file)} {command}"
            print(f"shellsmith: upload later with {command}", flush=True)
        self.replay()
        self.worker = threading.Thread(target=self._work, daemon=True,
                                       name="training-telemetry")
        self.worker.start()
        self.collector = threading.Thread(target=self._collect, daemon=True,
                                          name="training-observation-collector")
        self.collector.start()

    def phase(self, phase: str, *, outcome: str = "running") -> None:
        """Persist a trainer transition independently of SDK publication."""
        if self.observations is None:
            return
        try:
            self.enqueue("observation", self.observations.set_phase(phase, outcome=outcome))
        except Exception as error:
            self.collector_errors += 1
            self._warn(error)

    def _collect(self) -> None:
        """Collect CPU observations even when SDK publication has stalled."""
        while not self.stop.wait(self.config.telemetry.system_interval_seconds):
            self._health()

    def _publish_observation(self, record: dict) -> None:
        """Journal an observation once, then refresh bounded operator figures."""
        identity = "observation:" + record["id"]
        with self.lock:
            if identity in self.seen:
                return
            values = {key: value for key, value in record.items() if "/" in key}
            values.update(record.get("metrics", {}))
            values["run/phase"] = record["phase"]
            values["run/source_id"] = record.get("counters", {}).get("session_id") if record.get("counters") else "trainer"
            self.run.log(values)
            for progress in record.get("compute_records", []):
                self.run.log(progress)
            self._remember(identity)
            refresh = record["kind"] in {"phase", "update", "publication"} or time.monotonic() >= self.next_figures
            if refresh:
                self.run.summary.update(build_panels(self.observations, self.sdk,
                                                     plots=self.provider == "backprop"))
                self.next_figures = time.monotonic() + 60

    def enqueue(self, kind: str, *values: Any) -> None:
        """
        Purpose: Submit telemetry without a disk or network wait on the caller.
        Input: A publication kind and its durable source references.
        Output: No value. A full queue increments the deferred event counter.
        Note: Update and episode sources remain available for replay.
        """
        if self.run is None:
            return
        try:
            self.queue.put_nowait((kind, values))
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        """
        Purpose: Publish bounded batches outside the compute and rollout threads.
        Input: The publication queue and the stop event.
        Output: SDK events and a local live-health journal.
        """
        next_replay = time.monotonic() + 60
        while not self.stop.is_set() or not self.queue.empty():
            try:
                kind, values = self.queue.get(timeout=0.2)
            except queue.Empty:
                kind = None
            if kind is not None:
                try:
                    if kind == "observation":
                        self._publish_observation(*values)
                    elif kind == "progress":
                        self._publish_progress(*values)
                    elif kind == "publication":
                        self.publication(*values)
                    elif kind == "update":
                        self.update(*values)
                    else:
                        self.episode(*values)
                except Exception as error:
                    self._warn(error)
                finally:
                    self.queue.task_done()
            if time.monotonic() >= next_replay and not self.stop.is_set():
                try:
                    if self.provider == "backprop" and getattr(self.run, "_error", None) is not None:
                        self.run.flush()
                    pending = [record for record in list(self.observations.records)
                               if "observation:" + record["id"] not in self.seen]
                    for record in pending[:128]:
                        self._publish_observation(record)
                except Exception as error:
                    self._warn(error)
                next_replay = time.monotonic() + 60

    def _health(self) -> None:
        """
        Purpose: Persist live CPU counters without an operation on the GPU owner.
        Input: The configured Studio endpoint and local SDK journal.
        Output: A health record in the local journal and the selected sink.
        """
        if self.observations is None:
            return
        try:
            values = {"telemetry/queue_depth": self.queue.qsize(),
                      "telemetry/deferred_events": self.dropped,
                      "telemetry/logging_errors": self.errors,
                      "telemetry/collector_errors": self.collector_errors,
                      "telemetry/alive_timestamp": time.time()}
            with self.counter_lock:
                values["gm/completed_turn_calls_segment"] = self.gm_calls
                values["gm/turn_wall_seconds_segment"] = self.gm_seconds
            if self.provider == "backprop":
                values.update(self.sdk_health.read(self.run))
            live = None
            records = []
            if self.config.compute.backend == "studio":
                from studio_compute.client import Client
                try:
                    client_id = self.config.studio.training_client or self.config.run.run_id
                    live = Client(self.config.studio.endpoint).metrics(client_id=client_id)
                    values.update(_scalars({key: value for key, value in live.items()
                                           if key not in {"training_operations", "client_metrics"}}, "inference_service"))
                    values["inference_service/session_id"] = live.get("session_id")
                    values["compute/memory_peak_scope"] = "process_lifetime"
                    for operation_id, operation in live.get("training_operations", {}).items():
                        progress = operation.get("progress") or {}
                        values["compute/last_completed_phase"] = progress.get("phase")
                        records.append({**self.observations.context(),
                            "compute/memory_peak_scope": "process_lifetime",
                            "compute/operation_id": operation_id, "compute/kind": operation["kind"],
                            "compute/progress/phase": progress.get("phase"),
                            **_scalars(progress, "compute/progress")})
                except Exception as error:
                    self.collector_errors += 1
                    values["telemetry/collector_errors"] = self.collector_errors
                    values["telemetry/collector_error_type"] = type(error).__name__
            observation = self.observations.health(live, values, compute_records=records)
            records.insert(0, {**self.observations.context(), **values})
            with (self.root / "telemetry-health.jsonl").open("a") as stream:
                for record in records:
                    stream.write(_json(record) + "\n")
            self.enqueue("observation", observation)
        except Exception as error:
            self.collector_errors += 1
            self._warn(error)

    def _warn(self, error: Exception) -> None:
        self.errors += 1
        print(f"shellsmith: {self.label} logging failed ({type(error).__name__}: {error}); "
              "durable training logs retained for replay.", file=sys.stderr, flush=True)

    def _table(self, key: str, columns: list[str], rows: list[list]) -> Any:
        if key not in self.tables:
            self.tables[key] = self.sdk.Table(columns=columns, log_mode="INCREMENTAL")
            self.table_rows[key] = set()
        table = self.tables[key]
        for row in rows:
            identity = hashlib.sha256(_json(row).encode()).hexdigest()
            if key != "events/progress" and identity in self.table_rows[key]:
                continue  # a failed publication may already have staged this row
            table.add_data(*row)
            if key != "events/progress":
                self.table_rows[key].add(identity)
        return table

    def _remember(self, identity: str) -> None:
        if self.provider == "wandb":
            # W&B's log() enqueues work in its service. A crash before finish
            # must replay the source rather than trusting an unflushed queue.
            self.pending_receipts.add(identity)
            self.seen.add(identity)
            return
        with self.receipts.open("a") as handle:
            handle.write(identity + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.seen.add(identity)

    def progress(self, event) -> None:
        if not event.episode_id.startswith(self.config.run.run_id + "-s"):
            return
        if self.observations is not None:
            self.observations.progress(event)
        if event.phase == "turn_done":
            with self.counter_lock:
                self.gm_calls += 1
                self.gm_seconds += event.elapsed_seconds or 0.0
        # Coalesce repetitive live signals; terminal errors always pass.
        key = (event.episode_id, event.phase)
        now = time.monotonic()
        if "error" not in event.phase and now < self.next_progress:
            return
        self.next_progress = now + self.config.telemetry.system_interval_seconds
        if "error" not in event.phase and now - self.last_progress.get(key, -100) < 5:
            return
        self.last_progress[key] = now
        if len(self.last_progress) > 4096:
            self.last_progress.clear()
        self.enqueue("progress", event)

    def _publish_progress(self, event) -> None:
        if self.run is None or not event.episode_id.startswith(self.config.run.run_id + "-s"):
            return
        with self.lock:
            try:
                self.run.log({**self.observations.context(), "events/progress": self._table("events/progress",
                    ["episode_id", "phase", "turn", "elapsed_seconds", "message", "details"],
                    [[event.episode_id, event.phase, event.step_index, event.elapsed_seconds,
                      event.message, _json(asdict(event))]])})
                self.tables.pop("events/progress", None)
            except Exception as error:
                self._warn(error)

    def update(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        identity = "metric:" + hashlib.sha256(_json(row).encode()).hexdigest()
        with self.lock:
            if identity in self.seen:
                return
            try:
                values = {**self.observations.context(), "train/step": row["step"], "telemetry/segment": row.get("segment"),
                          "train/update_id": identity, "train/update": _json(row)}
                values.update(_scalars({key: value for key, value in row.items()
                                       if not key.endswith("_metrics") and key != "step"}, "train"))
                for key, value in row.items():
                    if key.endswith("_metrics") and isinstance(value, dict):
                        values.update(_scalars(value, key.removesuffix("_metrics")))
                total = len(self.snapshot.get("worlds", []))
                completed = row.get("next_world_offset", 0) + len(row.get("settled_world_offsets", []))
                values["train/worlds_completed"] = completed
                values["train/worlds_remaining"] = max(0, total - completed)
                values["train/optimizer_updates_committed"] = row.get("optimizer_updates_committed", self.state["rft_updates_completed"])
                observed = dict(row)
                observed["optimizer_updates_committed"] = values["train/optimizer_updates_committed"]
                if identity not in self.observations.updates:
                    self.enqueue("observation", self.observations.append("update", update_id=identity, update=observed))
                if completed > self.restored_offset:
                    values["train/eta_seconds_estimate"] = (
                        (time.monotonic() - self.started) / (completed - self.restored_offset)
                        * max(0, total - completed))
                worlds = row.get("worlds", [])
                rewards = [reward for world in worlds for reward in world.get("rewards", [])
                           if isinstance(reward, (int, float)) and math.isfinite(reward)]
                values.update({"train/worlds_collected": len(worlds),
                               "train/rollouts": sum(world.get("attempted_total", world.get("attempts", 0)) for world in worlds),
                               "train/accepted": sum(world.get("accepted", 0) for world in worlds),
                               "train/discarded": sum(world.get("discarded", 0) for world in worlds),
                               "train/failed": sum(world.get("failed", 0) for world in worlds),
                               "train/rejected": sum(world.get("rejected", 0) for world in worlds),
                               "train/skipped": int("skipped" in row),
                               "train/skipped_worlds": sum(world.get("decision") == "skip_insufficient" for world in worlds)})
                if values["train/rollouts"]:
                    values["train/acceptance_rate"] = values["train/accepted"] / values["train/rollouts"]
                if rewards:
                    mean = sum(rewards) / len(rewards)
                    values.update({"train/reward_mean": mean,
                                   "train/reward_std": math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards)),
                                   "train/reward_min": min(rewards), "train/reward_max": max(rewards)})
                if worlds:
                    values["train/worlds"] = self._table("train/worlds",
                        ["update_id", "step", "world_id", "decision", "details"],
                        [[identity, row["step"], world["world_id"], world["decision"], _json(world)]
                         for world in worlds])
                self.run.log(values)
                self.tables.clear()
                self.table_rows.clear()
                self._remember(identity)
            except Exception as error:
                self._warn(error)

    def episode(self, root: Path, outcome: dict) -> None:
        if self.run is None:
            return
        # A later provider discard withdraws acceptance. Publish that revision
        # under the same episode ID instead of retaining the earlier claim.
        identity = "episode:" + outcome["episode_id"] + ":" + hashlib.sha256(_json({
            "outcome": outcome, "trajectories": self.config.telemetry.log_trajectories,
        }).encode()).hexdigest()
        with self.lock:
            if identity in self.seen:
                return
            try:
                reward = ((outcome.get("audit") or {}).get("final_task") or {}).get("reward")
                episode_prefix = self.config.run.run_id + "-s"
                segment_text = (
                    outcome["episode_id"].removeprefix(episode_prefix).partition("-")[0]
                )
                source_segment = (int(segment_text)
                                  if outcome["episode_id"].startswith(episode_prefix)
                                  and segment_text.isdecimal() else None)
                values = {**self.observations.context(), "telemetry/segment": source_segment,
                          "rollout/episode_id": outcome["episode_id"],
                          **_scalars(outcome, "rollout")}
                seconds = outcome.get("wall_seconds", 0)
                if seconds > 0:
                    values["rollout/completion_tokens_per_second"] = (
                        outcome.get("token_usage", {}).get("completion_tokens", 0) / seconds
                    )
                if isinstance(reward, (int, float)) and math.isfinite(reward):
                    values["rollout/reward"] = reward
                values["rollout/attempted"] = 1
                values["rollout/accepted"] = int(bool(outcome.get("accepted_count")))
                values["rollout/failed"] = int(bool(outcome.get("error")))
                is_discarded = bool(outcome.get("discarded") or outcome.get("discarded_by_provider_fallback"))
                values["rollout/discarded"] = int(is_discarded)
                values["rollout/rejected"] = int(not outcome.get("accepted_count") and not outcome.get("error") and not is_discarded)
                stage = "discarded" if is_discarded else "failed" if outcome.get("error") else "accepted" if outcome.get("accepted_count") else "rejected"
                episode = {"terminal": True, "stage": stage, "world_id": outcome["world_id"],
                           "reward": reward, "failure_category": outcome.get("failure_category") or outcome.get("error_type") or ("unknown" if stage == "failed" else None)}
                if self.observations.episodes.get(outcome["episode_id"]) != episode:
                    self.enqueue("observation", self.observations.append("episode", episode_id=outcome["episode_id"], episode=episode))
                if self.config.telemetry.log_trajectories:
                    from shellsmith.viewer.loader import load_run

                    trace, error = load_run(root.parent, root.name)
                    trace = trace or {"artifact_errors": [error]}
                    # Includes failed/unaccepted generations never harvested into
                    # a trainable capture. PlayerEndpoint persists these first.
                    segment_root = next(parent for parent in root.parents
                                        if parent.parent == self.root / "segments")
                    records_path = segment_root / "player_records" / (outcome["episode_id"] + ".jsonl")
                    trace["model_requests"] = []
                    if records_path.is_file():
                        with records_path.open() as records:
                            for line in records:
                                if line.strip():
                                    record = json.loads(line)
                                    record.pop("payload", None)  # cumulative prompt repeated on every request
                                    trace["model_requests"].append(_compact(record))
                    values["rollouts/trajectories"] = self._table("rollouts/trajectories",
                        ["episode_id", "world_id", "policy_ref", "accepted", "reward", "error", "trajectory"],
                        [[outcome["episode_id"], outcome["world_id"], outcome.get("policy_ref"),
                          bool(outcome.get("accepted_count")), reward, outcome.get("error"),
                          _trajectory_json({"outcome": outcome, "trace": trace})]])
                self.run.log(values)
                self.tables.clear()
                self.table_rows.clear()
                self._remember(identity)
            except Exception as error:
                self._warn(error)

    def publication(self, row: dict) -> None:
        """Publish a durable policy receipt with a stable identity for resume."""
        identity = "publication:" + hashlib.sha256(_json(row).encode()).hexdigest()
        with self.lock:
            if self.run is None or identity in self.seen:
                return
            try:
                if identity not in self.observations.publications:
                    self.enqueue("observation", self.observations.append("publication", publication_id=identity, publication=row))
                self.run.log({**self.observations.context(), "train/step": row["step"], "publication/id": identity,
                              "publication/policy_ref": row["policy_ref"],
                              "publication/kind": row.get("kind", "policy"),
                              "publication/context": row.get("context", "update"),
                              **_scalars(row, "publication")})
                self._remember(identity)
            except Exception as error:
                self._warn(error)

    def wrap_runner(self, runner):
        async def tracked(job, args):
            outcome = await runner(job, args)
            self.enqueue("episode", job.output_dir, outcome)
            return outcome
        return tracked if self.run is not None else runner

    def replay(self) -> None:
        if self.run is None:
            return
        try:
            for record in list(self.observations.records):
                self._publish_observation(record)
            metrics = self.root / "metrics.jsonl"
            if metrics.is_file():
                for line in metrics.read_text().splitlines():
                    if line:
                        self.update(json.loads(line))
            for path in sorted((self.root / "policy-publications").glob("*.json")):
                self.publication(json.loads(path.read_text()))
            for path in sorted((self.root / "segments").glob("*/**/logs/durable_outcome.json")):
                root = path.parent.parent
                outcome = load_durable_outcome(root)
                if outcome is not None:
                    self.episode(root, outcome)
        except Exception as error:
            self._warn(error)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is None:
            return
        remove_listener(self.progress)
        self.phase("failed" if exc_type or self.state["status"] == "failed" else self.state["status"],
                   outcome="failed" if exc_type else self.state["status"])
        self._health()
        self.stop.set()
        if self.collector is not None:
            self.collector.join(timeout=5)
        if self.worker is not None:
            self.worker.join(timeout=60)
            if self.worker.is_alive():
                self._warn(TimeoutError("Telemetry worker remains blocked; durable sources retained."))
                return
        self.replay()
        try:
            self.run.summary.update(build_panels(self.observations, self.sdk,
                                                 plots=self.provider == "backprop"))
            self.run.summary.update({
                "trainer/status": "failed" if exc_type else self.state["status"],
                "trainer/step": self.state["step"],
                "trainer/rft_updates_completed": self.state["rft_updates_completed"],
                "trainer/next_world_offset": self.state["next_world_offset"],
                "trainer/segments": len(self.state["segments"]),
                "trainer/halt": _compact(self.state.get("halt")),
                "telemetry/logging_errors": self.errors,
                "telemetry/deferred_events": self.dropped,
                "gm/completed_turn_calls_segment": self.gm_calls,
                "gm/turn_wall_seconds_segment": self.gm_seconds,
                "telemetry/session_seconds": time.monotonic() - self.started,
            })
            if exc_value is not None:
                self.run.summary["trainer/error"] = str(exc_value)
                self.run.summary["trainer/error_type"] = exc_type.__name__
        except Exception as error:
            self._warn(error)
        finally:
            try:
                self.run.finish(exit_code=1 if exc_type or self.state["status"] == "failed" else 0)
                if self.pending_receipts:
                    with self.receipts.open("a") as handle:
                        handle.writelines(identity + "\n" for identity in sorted(self.pending_receipts))
                        handle.flush()
                        os.fsync(handle.fileno())
                    self.pending_receipts.clear()
            except Exception as error:
                self._warn(error)


def _trajectory_json(value: Any) -> str:
    """
    Purpose: Bound the final UTF-8 JSON row, including the preview envelope.
    Input: A trajectory with text and assessment data.
    Output: Sanitized JSON with at most MAX_TRAJECTORY_BYTES bytes.
    Method: Use binary search to find the longest preview that fits the envelope.
    """
    text = _json(value)
    size_bytes = len(text.encode("utf-8"))
    if size_bytes <= MAX_TRAJECTORY_BYTES:
        return text
    envelope = {"truncated": True, "original_characters": len(text),
                "original_bytes": size_bytes, "preview": ""}
    result = _json(envelope)
    low, high = 0, min(len(text), MAX_TRAJECTORY_BYTES)
    while low <= high:
        middle = (low + high) // 2
        candidate = _json(envelope | {"preview": text[:middle]})
        if len(candidate.encode("utf-8")) <= MAX_TRAJECTORY_BYTES:
            result = candidate
            low = middle + 1
        else:
            high = middle - 1
    return result
