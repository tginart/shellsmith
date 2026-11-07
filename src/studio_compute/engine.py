"""Single-owner resident engine. Client and policy factors share one frozen base."""

from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
import json
import math
import os
import resource
import subprocess
import time

from .model import Qwen27Model
from .protocol import digest
from .storage import file_hash

# prompt_groups: share complete prompt segments only among identical prompts.
# segment_trie: share every complete segment with an identical token prefix.
REPLAY_LAYOUTS = ("prompt_groups", "segment_trie")
# Trie gradient sums: exact balanced order first, then fewer retained partials.
TREE_SUM_LEVELS = (None, 4, 3, 2, 1)
# kern.memorystatus_level is the percent of host memory available. The probe
# that reproduced the perf4 hang stopped replay at this level.
MIN_MEMORY_LEVEL = 8
MEMORY_LEVEL_INTERVAL_S = 1.0
# Training replay frees and allocates group-VJP buffers of about 10 GB per group.
# With the 2 GiB service cache, MLX returns most of them to the host and the
# next group maps and faults them again (perf4 subset: 134 s; 8, 16, and 32 GiB
# all ~121 s). MLX frees cached buffers before active plus cached memory passes
# its limit, so admission and the replay guard do not change.
REPLAY_METAL_CACHE_BYTES = 8 << 30
# Replay uses each layer's weights once per segment and checkpoint group. Without
# a wired limit, the GPU makes idle buffers resident again at each use: a group
# VJP took 0.37-0.50 s in a five-group rotation against 0.32-0.41 s repeated,
# and a lookup in the ~50 GB PLE table took ~0.45 s. With the MLX residency set
# at the recommended working set (as mlx-lm generation does), the rotation
# costs nothing. Replay restores the previous limit afterwards.

# B: request batch, V: vocabulary, S: scalar.


def loss_metrics(kind, value):
    if kind not in ("cispo", "weighted_cross_entropy"):
        raise ValueError("Unsupported loss metric kind.")
    return {"loss": value, "loss_kind": kind, kind + "_loss": value}


class ClientAdapter:
    """Independent factors, selected by reference on the one resident decoder."""

    def __init__(self, model, factors):
        self.model, self.factors = model, factors

    def state(self):
        return self.factors

    def load(self, values):
        self.model.adapter.validate(values)
        self.factors = {
            name: value.astype(self.model.mx.bfloat16) for name, value in values.items()
        }
        self.model.mx.eval(self.factors)
        if any(
            not bool(self.model.mx.isfinite(value).all().item())
            for value in self.factors.values()
        ):
            raise FloatingPointError("Prospective BF16 adapter is nonfinite.")
        self.model.select(self.factors)


class ResidentEngine:
    def __init__(
        self,
        model_path,
        *,
        headroom_gib=16,
        allocator_gib=None,
        checkpoint_layers=None,
        segment_tokens=2048,
        head_rows=128,
        prefill_tokens=2048,
        metal_cache_gib=2,
        prefix_cache_gib=2,
        model_factory=Qwen27Model,
        execution_profile="bf16-fast",
        activation_budget_gib=None,
        replay_layout="prompt_groups",
        training_moe_chunk=0,
        prefill_moe_chunk=0,
    ):
        if execution_profile not in ("legacy", "bf16-fast"):
            raise ValueError("Unknown execution profile.")
        if replay_layout not in REPLAY_LAYOUTS:
            raise ValueError("Unknown replay layout.")
        self.replay_layout = replay_layout
        self.execution_profile = execution_profile
        self.resident = model_factory(
            model_path, headroom_gib=headroom_gib, allocator_gib=allocator_gib
        )
        self.mx, self.lm = self.resident.mx, self.resident.lm
        self.replay_ops = getattr(self.resident, "replay_ops", None)
        if type(training_moe_chunk) is not int or training_moe_chunk < 0:
            raise ValueError("Training MoE chunk must be zero or a positive integer.")
        if training_moe_chunk and self.replay_ops is None:
            raise ValueError("Grouped training experts require the Flash replay ops.")
        if training_moe_chunk:
            self.replay_ops.training_moe_chunk = training_moe_chunk
        if type(prefill_moe_chunk) is not int or prefill_moe_chunk < 0:
            raise ValueError("Prefill MoE chunk must be zero or a positive integer.")
        if prefill_moe_chunk and self.replay_ops is None:
            raise ValueError("Grouped prefill experts require the Flash replay ops.")
        if prefill_moe_chunk:
            self.resident.prefill_moe_chunk = prefill_moe_chunk
        self.saved_activation_backward = getattr(
            self.resident, "supports_saved_activations", True
        )
        if replay_layout == "segment_trie" and self.saved_activation_backward:
            raise ValueError("Segment-trie replay does not record saved activations.")
        if not math.isfinite(metal_cache_gib) or metal_cache_gib < 0:
            raise ValueError("Metal cache cap must be finite and nonnegative.")
        self.metal_cache_bytes = int(metal_cache_gib * 1024**3)
        self.mx.set_cache_limit(self.metal_cache_bytes)
        self.identity, self.capabilities = (
            self.resident.identity,
            self.resident.capabilities,
        )
        self.prefix_identity = digest(
            {
                "model": self.identity["model"],
                "numerics": self.identity.get("numerics"),
                "execution_profile": execution_profile,
                "kernel_build_id": self.identity.get("kernel_build_id"),
            }
        )
        self.compute_dtype = (
            self.mx.bfloat16 if execution_profile == "bf16-fast" else self.mx.float32
        )
        if execution_profile == "bf16-fast":
            self.capabilities = {
                **self.capabilities,
                "precision": {
                    **self.capabilities["precision"],
                    "optimizer": "bf16",
                    "recurrent_state": "bf16",
                    "gradients": "bf16",
                    "loss": "bf16",
                    "internal_accumulation": "native kernel registers",
                },
            }
        self.tokenizer_bundle = self.resident.bundle
        self.clients, self.policies = {}, OrderedDict()
        # Durable policy references outlive their disposable accelerator factors.
        self.policy_sources, self.policy_users = {}, {}
        self.policy_cache_limit = 4
        self.idle_factors = self.resident.adapter.state()
        self.checkpoint_override = checkpoint_layers
        if checkpoint_layers is None:
            checkpoint_layers = 8 if execution_profile == "legacy" else 1
        self.checkpoint_layers = checkpoint_layers
        if activation_budget_gib is not None and (
            isinstance(activation_budget_gib, bool)
            or not math.isfinite(activation_budget_gib)
            or activation_budget_gib < 0
        ):
            raise ValueError(
                "Activation budget must be auto or finite nonnegative GiB."
            )
        self.activation_budget_gib = activation_budget_gib
        self.projection_costs = {}
        if any(
            type(x) is not int or x < 1
            for x in (segment_tokens, head_rows, prefill_tokens)
        ):
            raise ValueError(
                "Replay segment, vocabulary tile and prefill sizes must be positive integers."
            )
        if prefill_tokens > 2048:
            raise ValueError(
                "Prefill chunks above 2048 exceed the current scratch admission estimate."
            )
        self.segment_tokens, self.head_rows, self.prefill_tokens = (
            segment_tokens,
            head_rows,
            prefill_tokens,
        )
        if type(checkpoint_layers) is not int or not 1 <= checkpoint_layers <= len(
            self.lm.layers
        ):
            raise ValueError("Invalid checkpoint layer group size.")
        self.identity = {
            **self.identity,
            "execution": {
                "execution_profile": execution_profile,
                "segment_tokens": segment_tokens,
                "head_rows": head_rows,
                "checkpoint_layers": (
                    "auto"
                    if self.checkpoint_override is None
                    and execution_profile == "bf16-fast"
                    else checkpoint_layers
                ),
                "activation_budget_gib": activation_budget_gib,
                "prefill_tokens": prefill_tokens,
                # The original layout keeps its original identity.
                **(
                    {"replay_layout": replay_layout}
                    if replay_layout != "prompt_groups"
                    else {}
                ),
                **(
                    {"training_moe_chunk": training_moe_chunk}
                    if training_moe_chunk
                    else {}
                ),
                **(
                    {"prefill_moe_chunk": prefill_moe_chunk}
                    if prefill_moe_chunk
                    else {}
                ),
            },
        }
        self.capabilities = {
            **self.capabilities,
            "execution": {
                "execution_profile": execution_profile,
                "segment_tokens": segment_tokens,
                "head_rows": head_rows,
                "checkpoint_layers": checkpoint_layers,
                "prefill_tokens": prefill_tokens,
                "metal_cache_limit_bytes": int(metal_cache_gib * 1024**3),
                "kernel_build_id": self.identity.get(
                    "kernel_build_id", self.identity.get("numerics")
                ),
                "activation_budget_gib": activation_budget_gib,
                "saved_activation_backward": execution_profile == "bf16-fast"
                and self.saved_activation_backward,
                "replay_layout": replay_layout,
                "training_moe_chunk": training_moe_chunk,
                "prefill_moe_chunk": prefill_moe_chunk,
            },
        }
        from .cache import PrefixCache

        self.prefixes = PrefixCache(self.mx, int(prefix_cache_gib * 1024**3))
        self.reserved = {}
        from .telemetry import memory, process_identity

        self.host_at_start, self.process = memory(), process_identity()

    def create(self, client, body):
        from .numerics.optimizer import MasterAdam

        seed = body.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Client seed must be a uint32.")
        if body.get("lora_rank", 16) != 16 or body.get("lora_scale", 2.0) != 2.0:
            raise ValueError("This model implementation supports rank 16, scale 2.")
        factor_bytes = sum(x.nbytes for x in self.resident.adapter.state().values())
        self.admit(factor_bytes * 7)
        adapter = ClientAdapter(self.resident, self.resident.adapter.initial(seed))
        self.mx.eval(adapter.state())
        optimizer = MasterAdam(
            adapter,
            body.get("learning_rate", 5e-5),
            **body.get("optimizer", {}),
            execution_profile=self.execution_profile,
            seed=seed,
        )
        self.mx.eval(optimizer.master, optimizer.first, optimizer.second)
        self.clients[client] = {
            "adapter": adapter,
            "optimizer": optimizer,
            "rng": self.mx.random.key(seed),
            "backward": None,
        }
        return {
            "adapter_rank": 16,
            "adapter_scale": 2.0,
            "rng_key": self.clients[client]["rng"].tolist(),
            "rng_counter": 0,
        }

    def optimizer_step_bytes(self, client):
        """Bound the optimizer's replacement tensors and profile-specific scratch."""
        entry = self.clients[client]
        optimizer = entry["optimizer"]
        master = sum(value.nbytes for value in optimizer.master.values())
        replacements = sum(
            value.nbytes
            for values in (optimizer.master, optimizer.first, optimizer.second)
            for value in values.values()
        )
        factors = sum(value.nbytes for value in entry["adapter"].state().values())
        if self.execution_profile == "bf16-fast":
            largest = max(
                (value.nbytes for value in optimizer.master.values()), default=0
            )
            # The fused kernel allocates three replacements. BF16 publication aliases master.
            return replacements + 2 * largest + 4096
        # Beyond the three replacement FP32 maps, first/second/master expressions
        # have 2/3/5 arithmetic intermediates per factor. Bound all ten maps even
        # when MLX fuses or releases them early, plus one factor for checks.
        scratch = 10 * master + max(
            (value.nbytes for value in optimizer.master.values()), default=0
        )
        return replacements + factors + scratch

    def restore_load_bytes(self, client):
        """Blank + loaded state, both BF16 casts, validation workspace and RNG keys."""
        factors = (
            self.clients[client]["adapter"].state()
            if client in self.clients
            else self.idle_factors
        )
        size = sum(value.nbytes for value in factors.values())
        largest = max((value.nbytes for value in factors.values()), default=0)
        # Loaded BF16 factors + three optimizer maps: FP32 maps make 7x the BF16
        # footprint, and bf16-fast BF16 maps make 4x. Allow two BF16 publication
        # casts and two largest-factor validation buffers. A cold restore still
        # creates a blank client of the same size before loading.
        map_ratio = 1 if self.execution_profile == "bf16-fast" else 2
        client_bytes = (1 + 3 * map_ratio) * size
        blank = client_bytes if client not in self.clients else 0
        return blank + client_bytes + 2 * size + 2 * largest + 3 * 8

    def restore(self, client, folder, state):
        fresh = client not in self.clients
        reservation = ("optimizer", client)
        held = self.reserved.pop(reservation, None)
        try:
            self._restore(client, folder, state)
        except Exception:
            if fresh:
                self.close_client(client)
            elif held is not None:
                self.reserved[reservation] = held
            raise

    def _restore(self, client, folder, state):
        mx = self.mx
        self.admit(self.restore_load_bytes(client))
        metadata = json.loads((folder / "optimizer.json").read_text())
        if client not in self.clients:
            self.create(
                client,
                {
                    "seed": state["seed"],
                    "learning_rate": metadata["learning_rate"],
                    "optimizer": {
                        "beta1": metadata["beta1"],
                        "beta2": metadata["beta2"],
                        "epsilon": metadata["eps"],
                    },
                },
            )
        entry = self.clients[client]
        values = mx.load(str(folder / "adapters.safetensors"))
        optimizer = mx.load(str(folder / "optimizer.safetensors"))
        if metadata["steps"] != state["revision"]:
            raise ValueError("Checkpoint revision differs from its optimizer count.")
        rng = mx.load(str(folder / "rng.safetensors"))
        entry["adapter"].model.adapter.validate(values)
        validated = entry["optimizer"].validated_state(optimizer, metadata)
        if any(
            not bool(
                mx.array_equal(
                    value, validated["master"][name].astype(mx.bfloat16)
                ).item()
            )
            for name, value in values.items()
        ):
            raise ValueError("Checkpoint BF16 factors differ from FP32 master.")
        if (
            set(rng) != {"key"}
            or rng["key"].shape != (2,)
            or rng["key"].dtype != mx.uint32
        ):
            raise ValueError("Invalid client RNG checkpoint.")
        entry["optimizer"].load(optimizer, metadata)
        # Small journaled sampling receipts may advance RNG since the last large
        # full-state checkpoint. Explicit restore uses that checkpoint's key.
        key = (
            mx.array(state["rng_key"], mx.uint32) if "rng_key" in state else rng["key"]
        )
        entry["rng"], entry["backward"] = key, None
        mx.eval(entry["adapter"].state(), entry["rng"])

    def write_state(self, client, folder):
        entry = self.clients[client]
        tensors, metadata = entry["optimizer"].state()
        mx = self.mx
        mx.save_safetensors(
            str(folder / "adapters.safetensors"), entry["adapter"].state()
        )
        mx.save_safetensors(str(folder / "optimizer.safetensors"), tensors)
        mx.save_safetensors(str(folder / "rng.safetensors"), {"key": entry["rng"]})
        (folder / "optimizer.json").write_text(json.dumps(metadata, sort_keys=True))

    def write_snapshot(self, client, folder, source):
        entry = self.clients[client]
        entry["optimizer"].state()  # reject snapshots with unconsumed gradients
        # Checkpoints are immutable. Hard links keep each full-state reference
        # self-contained without duplicating adapter/master/moment payloads.
        for file in source.iterdir():
            if file.name not in ("manifest.json", "rng.safetensors"):
                os.link(file, folder / file.name)
        self.mx.save_safetensors(str(folder / "rng.safetensors"), {"key": entry["rng"]})

    def policy(self, reference, folder, state, *, adapter_digest):
        # Retain the digest from the already verified publication manifest;
        # rereading changed files cannot redefine an immutable reference.
        if not isinstance(adapter_digest, str) or len(adapter_digest) != 64:
            raise ValueError(
                "Policy registration requires its published adapter digest."
            )
        source = (folder, state["revision"], adapter_digest)
        if (
            reference in self.policy_sources
            and self.policy_sources[reference][1:] != source[1:]
        ):
            raise ValueError("Published policy reference is immutable.")
        self.policy_sources.setdefault(reference, source)

    def _evict_policy(self):
        for reference in self.policies:
            if self.policy_users.get(reference, 0):
                continue
            policy = self.policies.pop(reference)
            if self.resident.selected is policy["factors"]:
                # Decoder modules themselves own the selected factors. Merely
                # deleting the cache entry would leave this allocation live.
                self.resident.select(self.idle_factors)
            return True
        return False

    @contextmanager
    def _borrow_policy(self, reference):
        if reference not in self.policies:
            folder, revision, expected = self.policy_sources[reference]
            asset = folder / "adapters.safetensors"
            if asset.is_symlink() or file_hash(asset) != expected:
                raise ValueError("Published policy adapter corruption.")
            while (
                len(self.policies) >= self.policy_cache_limit and self._evict_policy()
            ):
                pass
            self.admit(sum(x.nbytes for x in self.idle_factors.values()))
            factors = self.mx.load(str(folder / "adapters.safetensors"))
            self.resident.adapter.validate(factors)
            self.mx.eval(factors)
            self.policies[reference] = {"factors": factors, "revision": revision}
        self.policies.move_to_end(reference)
        self.policy_users[reference] = self.policy_users.get(reference, 0) + 1
        try:
            yield self.policies[reference]["factors"]
        finally:
            self.policy_users[reference] -= 1
            if not self.policy_users[reference]:
                del self.policy_users[reference]
            # Active operations may exceed the idle cache limit. Their exact
            # revisions stay pinned through every yield, error and cancellation.
            while len(self.policies) > self.policy_cache_limit and self._evict_policy():
                pass

    def metrics(self, started):
        from .telemetry import memory

        host = memory()
        return {
            **host,
            "swap_growth_bytes": host["swap_used_bytes"]
            - self.host_at_start["swap_used_bytes"],
            "process": self.process,
            "seconds": time.monotonic() - started,
            "allocator_active_bytes": self.mx.get_active_memory(),
            "allocator_peak_bytes": self.mx.get_peak_memory(),
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "swap": subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True
            ).strip(),
            "pid": os.getpid(),
            "precision": self.capabilities["precision"],
        }

    def next_rng(self, client):
        entry = self.clients[client]
        entry["rng"], draw_key = self.mx.random.split(entry["rng"])
        self.mx.eval(entry["rng"], draw_key)
        return entry["rng"].tolist(), draw_key.tolist()

    def persistent_bytes(self):
        arrays = {id(value): value for _, value in self.resident.frozen}
        arrays.update({id(value): value for value in self.idle_factors.values()})
        for entry in self.clients.values():
            optimizer = entry["optimizer"]
            for values in (
                entry["adapter"].state(),
                optimizer.master,
                optimizer.first,
                optimizer.second,
                optimizer.accumulated,
            ):
                arrays.update({id(value): value for value in values.values()})
            arrays[id(entry["rng"])] = entry["rng"]
        for policy in self.policies.values():
            arrays.update({id(value): value for value in policy["factors"].values()})
        return sum(value.nbytes for value in arrays.values()) + self.prefixes.bytes

    def admission_budget_bytes(self) -> int:
        """
        Purpose: Use one BF16 limit for replay, caches, and interleaved requests.
        Input: The allocator cap and the device recommended cap.
        Output: The smaller cap for BF16. Legacy keeps its allocator cap.
        """
        if self.execution_profile == "legacy":
            return self.resident.budget
        recommended = self.capabilities.get("recommended_working_set_bytes")
        return min(self.resident.budget, int(recommended or self.resident.budget))

    def admit(self, additional):
        while True:
            try:
                persistent = self.persistent_bytes()
                reserved = sum(self.reserved.values())
                if self.execution_profile == "bf16-fast":
                    needed = max(
                        self.mx.get_active_memory() + additional,
                        persistent + reserved + additional,
                    )
                    budget = self.admission_budget_bytes()
                    if needed > budget:
                        raise MemoryError(
                            f"Admission needs {needed} bytes; budget is {budget}."
                        )
                self.resident.admit(additional, reserved, persistent=persistent)
                return
            except MemoryError:
                if not self._evict_policy():
                    raise

    def _tree_replay(self, kind):
        return kind == "forward_backward" and self.replay_layout == "segment_trie"

    def _tree_head_bytes(self, groups):
        """Bound head rows that trie nodes retain until their own backward."""
        width = self.lm.args.hidden_size * getattr(self.lm.args, "hc_count", 1)
        rows = sum(
            sum(1 for value in (sequence.get("loss_mask")
                                or [1] * len(sequence["sampled_tokens"])) if value)
            for group in groups.values()
            for sequence in group
        )
        return rows * (width * 2 + 12)

    def _tree_admission(self, sequences, future_gradient_bytes):
        """
        Purpose: Admit one trie replay with the adapter-gradient copies it keeps.
        Input: The replayed sequences in branch order. The bytes of one gradient.
        Output: The interval, peak reservation, optional byte caps, and sum level cap.
        Method:
            1. Price each sequence path with estimate_replay, as before.
            2. Add the head rows and the peak count of gradient copies in run_tree.
            3. Try the exact balanced sums first. If no interval fits, cap the levels.
        Note: A level cap changes only the BF16 rounding order of the gradient sums.
        """
        from .numerics.replay import tree_gradient_copies

        token_ids = [[*value["prompt_token_ids"], *value["sampled_tokens"][:-1]]
                     for value in sequences]
        paths = {(index,): [value] for index, value in enumerate(sequences)}
        head_bytes = self._tree_head_bytes({None: sequences})
        failure = None
        for levels in TREE_SUM_LEVELS:
            copies = tree_gradient_copies(token_ids, self.segment_tokens, levels)
            try:
                interval, peak, budgets = self._saved_admission(
                    paths, future_gradient_bytes, gradient_copies=copies,
                    retained_bytes=head_bytes)
            except MemoryError as error:
                failure = error
                continue
            return interval, peak, budgets, levels
        raise MemoryError(f"No trie gradient sum level fits: {failure}")

    def _saved_admission(self, groups, future_gradient_bytes, *, gradient_copies=2,
                         retained_bytes=0):
        """
        Purpose: Choose the smallest checkpoint interval that admits mandatory state.
        Input: Active replay groups and reserved future gradients. The gradient
            copies that replay keeps, and other bytes that replay retains.
        Output: The interval, peak reservation, and per-prompt optional byte caps.
        Note: Explicit intervals and activation limits fail when memory is insufficient.
        """
        from .activation_budget import SAFETY_BYTES, activation_budget
        from .numerics.retention import mandatory_saved_bytes

        intervals = (
            [self.checkpoint_override]
            if self.checkpoint_override is not None
            else range(1, len(self.lm.layers) + 1)
        )
        persistent = max(
            self.mx.get_active_memory(),
            self.persistent_bytes() + sum(self.reserved.values()),
        )
        recommended = (
            self.capabilities.get("recommended_working_set_bytes")
            or self.resident.budget
        )
        scratch = future_gradient_bytes * gradient_copies + retained_bytes
        failure = None
        for interval in intervals:
            budgets, peaks = {}, []
            try:
                for prompt, group in groups.items():
                    length = max(
                        len(value["prompt_token_ids"])
                        + len(value["sampled_tokens"])
                        - 1
                        for value in group
                    )
                    estimate = self.resident.estimate_replay(
                        group,
                        interval,
                        segment_tokens=self.segment_tokens,
                        head_rows=self.head_rows,
                    )["additional_bytes"]
                    mandatory = (
                        mandatory_saved_bytes(self.lm.layers, length)
                        if self.saved_activation_backward
                        else 0
                    )
                    budgets[prompt] = activation_budget(
                        allocator_bytes=self.resident.budget,
                        recommended_bytes=int(recommended),
                        resident_bytes=int(persistent),
                        checkpoint_bytes=estimate + mandatory,
                        scratch_bytes=scratch,
                        requested_gib=(
                            self.activation_budget_gib
                            if self.saved_activation_backward
                            else 0
                        ),
                    )
                    peaks.append(
                        estimate
                        + mandatory
                        + budgets[prompt]
                        + scratch
                        + SAFETY_BYTES
                    )
            except MemoryError as error:
                failure = error
                if self.checkpoint_override is not None:
                    raise
                continue
            return interval, max(peaks, default=0), budgets
        raise MemoryError(f"No full-context checkpoint interval fits: {failure}")

    def _embed(self, ids_BL):
        """Return the BF16 replay input rows for token IDs. Each row depends on its ID only."""
        embeddings = self.lm.model.embed_tokens(ids_BL).astype(self.mx.bfloat16)
        if self.replay_ops is not None:
            embeddings = self.replay_ops.expand(self.lm, embeddings)
        return embeddings

    def _check_replay_memory(self, level_checked_s: float) -> float:
        """
        Purpose: Stop a replay before its memory can stall the host.
        Input: The monotonic time of the last host memory level check.
        Output: The time of the latest level check. Excess memory raises MemoryError.
        Note: The cap is the admission budget. Admission plans replay below it,
            so the guard only stops an estimate error. The service then aborts
            the operation, and the arrays go free with the replay generator.
        """
        active = self.mx.get_active_memory()
        cap = self.admission_budget_bytes()
        if active > cap:
            raise MemoryError(f"Replay memory {active} bytes exceeds the {cap} byte cap.")
        now = time.monotonic()
        if now - level_checked_s < MEMORY_LEVEL_INTERVAL_S:
            return level_checked_s
        level = _memory_level()
        if level < MIN_MEMORY_LEVEL:
            raise MemoryError(
                f"Host memory level {level} is below {MIN_MEMORY_LEVEL} during replay."
            )
        return now

    def work(self, client, kind, body):
        mx = self.mx
        started = time.monotonic()
        if kind in ("sample", "logprobs"):
            return (yield from self._sample(body, started))
        entry = self.clients[client]
        if kind == "optim_step":
            if entry["backward"] != body.get("backward_operation"):
                raise ValueError(
                    "Optimizer must consume the matching live backward operation."
                )
            reservation = ("optimizer", client)
            held = self.reserved.pop(reservation, None)
            try:
                self.admit(self.optimizer_step_bytes(client))
                self.resident.select(entry["adapter"].state())
                norm = entry["optimizer"].step(learning_rate=body["learning_rate"])
            except Exception:
                if held is not None:
                    self.reserved[reservation] = held
                raise
            entry["backward"] = None
            return {
                "optim_metrics": {
                    "gradient_norm": norm,
                    "optimizer_steps": entry["optimizer"].steps,
                    "optimizer_committed": True,
                    "learning_rate": body["learning_rate"],
                },
                "execution_metrics": self.metrics(started),
            }
        if kind == "inspect":
            state_hashes = None
            if body.get("hash_state"):
                tensors, metadata = entry["optimizer"].state()
                state_hashes = {
                    "adapter": self.resident.tensor_digest(entry["adapter"].state()),
                    "optimizer": self.resident.tensor_digest(tensors),
                    "rng": self.resident.tensor_digest({"key": entry["rng"]}),
                    "optimizer_metadata": metadata,
                }
            return {
                "base_allocation": digest(self.resident.base_allocation),
                "frozen_base_hash": self.resident.frozen_digest()
                if body.get("hash_base")
                else None,
                "optimizer_steps": entry["optimizer"].steps,
                "rng": entry["rng"].tolist(),
                "state_hashes": state_hashes,
                "metrics": self.metrics(started),
            }
        sequences = body["sequences"]
        factors = entry["adapter"].state()
        if entry["backward"] is not None:
            raise ValueError(
                "Consume or abort the client's preceding backward operation first."
            )
        released_cache_bytes = 0
        if kind == "forward_backward":
            # PrefixCache.get returns independent working copies. Clearing idle
            # snapshots never touches caches held by an active generator.
            released_cache_bytes = self.prefixes.bytes
            self.prefixes.clear()
            mx.clear_cache()
        total_tokens = sum(
            len(s["prompt_token_ids"]) + len(s["sampled_tokens"]) - 1 for s in sequences
        )
        total_supervised = sum(
            sum(s.get("loss_mask", [1.0] * len(s["sampled_tokens"]))) for s in sequences
        )
        completed_segments = completed_tokens = completed_supervised = 0
        phase_seconds = {}
        groups = {}
        replay_groups = {}
        zero_gradient_indices = set()
        for index, sequence in enumerate(sequences):
            prompt = tuple(sequence["prompt_token_ids"])
            groups.setdefault(prompt, []).append((index, sequence))
            if (
                kind == "forward_backward"
                and sequence.get("source_spans")
                and not any(sequence["loss_mask"])
            ):
                zero_gradient_indices.add(index)
            else:
                replay_groups.setdefault(prompt, []).append(sequence)
        peak = max(
            (
                self.resident.estimate_replay(
                    group,
                    self.checkpoint_layers,
                    segment_tokens=self.segment_tokens,
                    head_rows=self.head_rows,
                )["additional_bytes"]
                for group in replay_groups.values()
            ),
            default=0,
        )
        replay_checkpoint_layers = self.checkpoint_layers
        activation_budgets = {}
        gradient_sum_levels = None
        if kind == "forward_backward":
            future_gradients = sum(
                value.nbytes for value in entry["optimizer"].master.values()
            )
            if self.execution_profile == "bf16-fast" and self._tree_replay(kind):
                (replay_checkpoint_layers, peak, activation_budgets,
                 gradient_sum_levels) = self._tree_admission(
                    [sequence for index, sequence in enumerate(sequences)
                     if index not in zero_gradient_indices],
                    future_gradients,
                )
            elif self.execution_profile == "bf16-fast":
                replay_checkpoint_layers, peak, activation_budgets = (
                    self._saved_admission(replay_groups, future_gradients)
                )
            peak = max(peak, self.optimizer_step_bytes(client) + future_gradients)
        if self._tree_replay(kind):
            groups = {None: list(enumerate(sequences))}
        self.admit(peak)
        entry["optimizer"].zero()
        entry["backward"] = body["operation_id"] if kind == "forward_backward" else None
        self.reserved[body["operation_id"]] = peak
        loss, logprobs, group_stats = 0.0, [None] * len(sequences), []
        wired_bytes = None
        if kind == "forward_backward":
            mx.set_cache_limit(max(self.metal_cache_bytes, REPLAY_METAL_CACHE_BYTES))
            recommended = self.capabilities.get("recommended_working_set_bytes")
            if recommended:
                wired_bytes = mx.set_wired_limit(int(recommended))
        try:
            from .numerics.cispo import cispo_loss
            from .numerics.replay import Replay

            for prompt, compatible in groups.items():
                self.resident.select(factors)
                branches = []
                active_compatible = []
                for sequence_index, sequence in compatible:
                    if sequence_index in zero_gradient_indices:
                        # A context-only segment has an exact zero gradient.
                        # Preserve a requested Adam update, including momentum.
                        entry["optimizer"].accumulate(
                            {
                                name: mx.zeros_like(value)
                                for name, value in entry["optimizer"].master.items()
                            }
                        )
                        logprobs[sequence_index] = []
                        continue
                    sparse = kind == "forward_backward" and bool(
                        sequence.get("source_spans")
                    )
                    selected = (
                        [i for i, m in enumerate(sequence["loss_mask"]) if m]
                        if sparse
                        else None
                    )
                    active_compatible.append((sequence_index, sequence))
                    sequence_prompt = tuple(sequence["prompt_token_ids"])
                    token_ids = [*sequence_prompt, *sequence["sampled_tokens"][:-1]]
                    ids = mx.array([token_ids], mx.int32)
                    embeddings = self._embed(ids)
                    positions = mx.broadcast_to(
                        mx.arange(ids.shape[1], dtype=mx.int32)[None, None],
                        (3, 1, ids.shape[1]),
                    )
                    objective = None
                    if kind == "forward_backward":
                        mask = sequence.get(
                            "loss_mask", [1.0] * len(sequence["sampled_tokens"])
                        )
                        if selected is not None:
                            sequence = {
                                **sequence,
                                **{
                                    key: [sequence[key][i] for i in selected]
                                    for key in (
                                        "behavior_logprobs",
                                        "advantages",
                                        "loss_mask",
                                        "weights",
                                    )
                                    if sequence.get(key) is not None
                                },
                            }
                            mask = sequence["loss_mask"]
                        if body.get("loss", "cispo") == "cispo":
                            objective = lambda current, s=sequence, m=mask: cispo_loss(
                                current,
                                s["behavior_logprobs"],
                                s["advantages"],
                                m,
                                clip_low=body.get("clip_low", 0.0),
                                clip_high=body.get("clip_high", 4.0),
                                dtype=self.compute_dtype,
                            )
                        else:
                            objective = lambda current, s=sequence, m=mask: (
                                -mx.sum(
                                    current.astype(self.compute_dtype)
                                    * mx.array(s["weights"], self.compute_dtype)
                                    * mx.array(m, self.compute_dtype)
                                )
                            )
                    branches.append(
                        {
                            "embeddings": embeddings,
                            "positions": positions,
                            "prefix_length": len(sequence_prompt),
                            "token_ids": token_ids,
                            "tokens": sequence["sampled_tokens"],
                            "objective": objective,
                            **(
                                {"input_ids": ids}
                                if self.replay_ops is not None
                                else {}
                            ),
                            **(
                                {"target_indices": selected}
                                if selected is not None
                                else {}
                            ),
                            # Trie replay embeds one segment at a time. A sliced
                            # whole-branch embedding stays resident to the end.
                            **(
                                {"segment_embeddings": lambda start, stop, ids=ids:
                                 self._embed(ids[:, start:stop])}
                                if self._tree_replay(kind)
                                else {}
                            ),
                        }
                    )
                replay = Replay(
                    self.resident.model,
                    checkpoint_layers=replay_checkpoint_layers,
                    segment_tokens=self.segment_tokens,
                    head_rows=self.head_rows,
                    execution_profile=self.execution_profile,
                    saved_activations=(
                        kind == "forward_backward"
                        and self.execution_profile == "bf16-fast"
                        and self.saved_activation_backward
                    ),
                    activation_budget_bytes=activation_budgets.get(prompt, 0),
                    projection_costs=self.projection_costs,
                    model_ops=self.replay_ops,
                    gradient_sum_levels=gradient_sum_levels,
                )
                if not branches:
                    work = iter(())
                elif self._tree_replay(kind):
                    work = replay.run_tree(branches)
                else:
                    work = replay.run_group(branches)
                level_checked_s = 0.0
                while True:
                    self.resident.select(factors)
                    self.resident.mode(kind == "forward_backward")
                    phase_started = time.monotonic()
                    try:
                        progress = next(work)
                    except StopIteration as result:
                        currents, value, gradients, stats = (
                            result.value if branches else ([], None, None, {})
                        )
                        break
                    if self.execution_profile == "bf16-fast":
                        level_checked_s = self._check_replay_memory(level_checked_s)
                    now = time.monotonic()
                    phase = progress["phase"]
                    phase_seconds[phase] = (
                        phase_seconds.get(phase, 0.0) + now - phase_started
                    )
                    yield {
                        **progress,
                        "completed_segments": completed_segments,
                        "total_segments": len(sequences),
                        "completed_tokens": completed_tokens,
                        "total_tokens": total_tokens,
                        "completed_supervised_tokens": completed_supervised,
                        "supervised_tokens": total_supervised,
                        "phase_seconds": dict(phase_seconds),
                        "optimizer_committed": False,
                        "released_prefix_cache_bytes": released_cache_bytes,
                        "prefix_cache_bytes": self.prefixes.bytes,
                        "active_memory_bytes": mx.get_active_memory(),
                        "peak_memory_bytes": mx.get_peak_memory(),
                    }
                for (index, _), current in zip(active_compatible, currents):
                    logprobs[index] = current.tolist()
                group_stats.append(stats)
                if value is not None:
                    loss += float(value.item())
                    entry["optimizer"].accumulate(gradients)
                completed_segments += len(compatible)
                completed_tokens += sum(
                    len(s["prompt_token_ids"]) + len(s["sampled_tokens"]) - 1
                    for _, s in compatible
                )
                completed_supervised += sum(
                    sum(s.get("loss_mask", [1.0] * len(s["sampled_tokens"])))
                    for _, s in compatible
                )
                yield {
                    "phase": "segment_complete",
                    "completed_segments": completed_segments,
                    "total_segments": len(sequences),
                    "completed_tokens": completed_tokens,
                    "total_tokens": total_tokens,
                    "completed_supervised_tokens": completed_supervised,
                    "supervised_tokens": total_supervised,
                    "phase_seconds": dict(phase_seconds),
                    "optimizer_committed": False,
                    "active_memory_bytes": mx.get_active_memory(),
                    "peak_memory_bytes": mx.get_peak_memory(),
                    "prefix_cache_bytes": self.prefixes.bytes,
                }
                del branches, currents, gradients
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite replay loss.")
            if kind == "forward_backward":
                entry["backward"] = body["operation_id"]
                # Keep the future optimizer peak owned between RPCs; caches and
                # interleaved clients must not consume it after replay finishes.
                self.reserved.pop(body["operation_id"], None)
                self.reserved[("optimizer", client)] = self.optimizer_step_bytes(client)
            return {
                "forward_metrics": {
                    **loss_metrics(body.get("loss", "cispo"), loss),
                    "reduction": "sum",
                    "sequences": len(sequences),
                    "completed_segments": completed_segments,
                    "total_segments": len(sequences),
                    "completed_tokens": completed_tokens,
                    "total_tokens": total_tokens,
                    "supervised_tokens": total_supervised,
                    "phase_seconds": phase_seconds,
                    "released_prefix_cache_bytes": released_cache_bytes,
                    "optimizer_committed": False,
                    "prompt_tokens": sum(len(x["prompt_token_ids"]) for x in sequences),
                    "completion_tokens": sum(
                        len(x["sampled_tokens"]) for x in sequences
                    ),
                    "loss_tokens": sum(
                        sum(x.get("loss_mask", [1.0] * len(x["sampled_tokens"])))
                        for x in sequences
                    ),
                },
                # Packed training outputs are sparse and explicitly indexed.
                "logprob_target_indices": [
                    (
                        [i for i, m in enumerate(s["loss_mask"]) if m]
                        if kind == "forward_backward" and s.get("source_spans")
                        else None
                    )
                    for s in sequences
                ],
                "logprobs": logprobs,
                "groups": group_stats,
                "execution_metrics": self.metrics(started),
            }
        finally:
            self.reserved.pop(body["operation_id"], None)
            self.resident.mode(False)
            mx.set_cache_limit(self.metal_cache_bytes)
            if wired_bytes is not None:
                mx.set_wired_limit(wired_bytes)

    def _sample(self, body, started):
        maximum = (
            len(body["target_token_ids"])
            if body.get("target_token_ids")
            else body["max_tokens"]
        )
        peak = self.resident.estimate_inference(
            len(body["prompt_token_ids"]) + maximum,
            cache_precision=body.get("cache_precision", "bf16"),
        )
        self.admit(peak)
        reservation = ("inference", body.get("operation_id", id(body)))
        self.reserved[reservation] = peak
        try:
            return (yield from self._sample_units(body, started))
        finally:
            self.reserved.pop(reservation, None)

    def set_inference_observer(self, observer):
        """Owner-thread CPU observer; called when an output token materializes."""
        self.inference_observer = observer

    def _observe_generated(self, body, count, finished):
        observer = getattr(self, "inference_observer", None)
        if observer is not None:
            observer(body.get("operation_id"), count, finished=finished)

    def _decode_hidden(self, ids, *, position_ids, cache):
        if self.replay_ops is not None:
            return self.resident.decode_hidden(ids, position_ids, cache)
        if self.execution_profile == "legacy":
            return self.lm.model(ids, position_ids=position_ids, cache=cache)
        from .numerics.bf16 import decode_hidden

        return decode_hidden(self.lm, ids, position_ids, cache)

    def _decode_logits(self, ids, *, position_ids, cache):
        if self.execution_profile == "legacy":
            return self.lm(ids, position_ids=position_ids, cache=cache)
        from types import SimpleNamespace

        hidden = self._decode_hidden(ids, position_ids=position_ids, cache=cache)
        decoder = self.lm.model
        head = (
            decoder.embed_tokens.as_linear
            if self.lm.args.tie_word_embeddings
            else self.lm.lm_head
        )
        return SimpleNamespace(logits=head(hidden))

    def _draw_token(self, logprobs_BV, key):
        """Draw from the unmodified categorical policy."""
        return self.mx.random.categorical(logprobs_BV, key=key)

    def _sample_units(self, body, started):
        with self._borrow_policy(body["policy"]) as factors:
            return (yield from self._sample_policy_units(body, started, factors))

    def _sample_policy_units(self, body, started, factors):
        mx = self.mx
        prompt = body["prompt_token_ids"]
        forced = body.get("target_token_ids")
        maximum = len(forced) if forced else body["max_tokens"]
        precision = body.get("cache_precision", "bf16")
        hit = (
            self.prefixes.get(self.prefix_identity, body["policy"], precision, prompt)
            if body.get("use_prefix_cache", True)
            else None
        )
        offset = 0
        if hit:
            cache, logits, offset = hit
        else:
            cache = self.lm.make_cache()
            if precision == "8bit":
                cache = [
                    x.to_quantized(bits=8, group_size=64)
                    if hasattr(x, "to_quantized")
                    else x
                    for x in cache
                ]
        ids = mx.array([prompt], mx.int32)
        decoder = self.lm.model
        head = (
            decoder.embed_tokens.as_linear
            if self.lm.args.tie_word_embeddings
            else self.lm.lm_head
        )
        for start in range(offset, len(prompt), self.prefill_tokens):
            self.resident.select(factors)
            self.resident.mode(False)
            stop = min(len(prompt), start + self.prefill_tokens)
            positions = mx.broadcast_to(
                mx.arange(start, stop, dtype=mx.int32)[None, None], (3, 1, stop - start)
            )
            hidden = self._decode_hidden(
                ids[:, start:stop], position_ids=positions, cache=cache
            )
            mx.eval(hidden, [x.state for x in cache])
            yield {"phase": "prefill", "tokens": stop, "total_tokens": len(prompt)}
        if offset < len(prompt):
            logits = head(hidden[:, -1:])[:, -1].astype(self.compute_dtype)
            mx.eval(logits)
            if body.get("use_prefix_cache", True):
                self.prefixes.put(
                    self.prefix_identity,
                    body["policy"],
                    precision,
                    prompt,
                    cache,
                    logits,
                )
        prefill_seconds = time.monotonic() - started
        decode_started = time.monotonic()
        key = (
            mx.array(body["rng_key"], mx.uint32)
            if "rng_key" in body
            else mx.random.key(body["seed"])
        )
        sampled, probabilities, reason = [], [], "length"
        for index in range(maximum):
            self.resident.select(factors)
            self.resident.mode(False)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if forced:
                token = forced[index]
            else:
                key, draw_key = mx.random.split(key)
                draw = self._draw_token(lp, draw_key)
                mx.eval(draw, key)
                token = int(draw.item())
            probability = float(lp[0, token].item())
            if not math.isfinite(probability):
                raise FloatingPointError("Nonfinite behavior logprob.")
            sampled.append(token)
            probabilities.append(probability)
            # Observe before evaluating the next decode step. If that step
            # fails, this already-materialized token remains measured work.
            if not forced:
                self._observe_generated(
                    body,
                    len(sampled),
                    token in body.get("stop_token_ids", []) or index + 1 == maximum,
                )
            if not forced and token in body.get("stop_token_ids", []):
                reason = "stop"
                yield {
                    "phase": "decode",
                    "tokens": len(sampled),
                    "generated_tokens": [len(sampled)],
                    "finished_requests": [True],
                }
                break
            if index + 1 < maximum:
                out = self._decode_logits(
                    mx.array([[token]], mx.int32),
                    position_ids=mx.full((3, 1, 1), len(prompt) + index, mx.int32),
                    cache=cache,
                )
                logits = (out.logits if hasattr(out, "logits") else out)[:, -1].astype(
                    self.compute_dtype
                )
                mx.eval(logits, [x.state for x in cache])
            yield {
                "phase": "decode",
                "tokens": len(sampled),
                "generated_tokens": [len(sampled)],
                "finished_requests": [index + 1 == maximum],
            }
        return {
            "sampled_tokens": sampled,
            "behavior_logprobs": probabilities,
            "prompt_tokens": len(prompt),
            "completion_tokens": len(sampled),
            "policy_revision": body["revision"],
            "policy": body["policy"],
            "finish_reason": reason,
            "cache_precision": precision,
            "prefix_cache_tokens": offset,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": time.monotonic() - decode_started,
            "logprobs": probabilities,
            "execution_metrics": self.metrics(started),
        }

    def abort(self, client, operation_id=None):
        if client in self.clients and (
            operation_id is None
            or self.clients[client]["backward"] in (None, operation_id)
        ):
            self.clients[client]["optimizer"].zero()
            self.clients[client]["backward"] = None
            self.reserved.pop(("optimizer", client), None)

    def sample_batch(self, bodies):
        peak = self.resident.estimate_inference(
            len(bodies[0]["prompt_token_ids"]) + bodies[0]["max_tokens"],
            len(bodies),
            cache_precision=bodies[0].get("cache_precision", "bf16"),
        )
        self.admit(peak)
        reservation = ("inference_batch", id(bodies))
        self.reserved[reservation] = peak
        try:
            return (yield from self._sample_batch_units(bodies))
        finally:
            self.reserved.pop(reservation, None)

    def _sample_batch_units(self, bodies):
        with self._borrow_policy(bodies[0]["policy"]) as factors:
            return (yield from self._sample_batch_policy_units(bodies, factors))

    def _sample_batch_policy_units(self, bodies, factors):
        """Native inference for up to four equal-length compatible requests."""
        mx, started = self.mx, time.monotonic()
        count, length, maximum = (
            len(bodies),
            len(bodies[0]["prompt_token_ids"]),
            bodies[0]["max_tokens"],
        )
        if not 1 <= count <= 4 or any(
            (
                x["policy"],
                len(x["prompt_token_ids"]),
                x["max_tokens"],
                x.get("cache_precision", "bf16"),
            )
            != (
                bodies[0]["policy"],
                length,
                maximum,
                bodies[0].get("cache_precision", "bf16"),
            )
            for x in bodies
        ):
            raise ValueError("Incompatible inference batch.")
        cache = self.lm.make_cache()
        precision = bodies[0].get("cache_precision", "bf16")
        if precision == "8bit":
            cache = [
                x.to_quantized(bits=8, group_size=64)
                if hasattr(x, "to_quantized")
                else x
                for x in cache
            ]
        ids = mx.array([x["prompt_token_ids"] for x in bodies], mx.int32)
        for start in range(0, length, self.prefill_tokens):
            self.resident.select(factors)
            self.resident.mode(False)
            stop = min(length, start + self.prefill_tokens)
            positions = mx.broadcast_to(
                mx.arange(start, stop, dtype=mx.int32)[None, None],
                (3, count, stop - start),
            )
            hidden = self._decode_hidden(
                ids[:, start:stop], position_ids=positions, cache=cache
            )
            mx.eval(hidden, [x.state for x in cache])
            yield {"phase": "batched_prefill", "tokens": stop, "batch_size": count}
        decoder = self.lm.model
        head = (
            decoder.embed_tokens.as_linear
            if self.lm.args.tie_word_embeddings
            else self.lm.lm_head
        )
        logits = head(hidden[:, -1:])[:, -1].astype(self.compute_dtype)
        mx.eval(logits)
        prefill_seconds = time.monotonic() - started
        decode_started = time.monotonic()
        keys = [
            mx.array(x["rng_key"], mx.uint32)
            if "rng_key" in x
            else mx.random.key(x["seed"])
            for x in bodies
        ]
        sampled, probabilities, finished = (
            [[] for _ in bodies],
            [[] for _ in bodies],
            [False] * count,
        )
        for index in range(maximum):
            self.resident.select(factors)
            self.resident.mode(False)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            drawn = [0] * count
            pending = []
            for row, body in enumerate(bodies):
                if finished[row]:
                    continue
                if body.get("target_token_ids"):
                    draw_S = mx.array(body["target_token_ids"][index], mx.int32)
                else:
                    keys[row], key = mx.random.split(keys[row])
                    draw_S = self._draw_token(lp[row], key)
                probability_S = mx.take(lp[row], draw_S, axis=-1)
                pending.append((row, draw_S, probability_S))
            mx.eval(
                [draw_S for _, draw_S, _ in pending],
                [probability_S for _, _, probability_S in pending],
                keys,
            )
            for row, draw_S, probability_S in pending:
                body = bodies[row]
                token = int(draw_S.item())
                probability = float(probability_S.item())
                if not math.isfinite(probability):
                    raise FloatingPointError("Nonfinite batch behavior logprob.")
                sampled[row].append(token)
                probabilities[row].append(probability)
                drawn[row] = token
                finished[row] = token in body.get("stop_token_ids", [])
                if not body.get("target_token_ids"):
                    self._observe_generated(
                        body, len(sampled[row]), finished[row] or index + 1 == maximum
                    )
            if all(finished):
                yield {
                    "phase": "batched_decode",
                    "tokens": index + 1,
                    "batch_size": count,
                    "generated_tokens": [len(tokens) for tokens in sampled],
                    "finished_requests": list(finished),
                }
                break
            if index + 1 < maximum:
                out = self._decode_logits(
                    mx.array(drawn, mx.int32)[:, None],
                    cache=cache,
                    position_ids=mx.full((3, count, 1), length + index, mx.int32),
                )
                logits = (out.logits if hasattr(out, "logits") else out)[:, -1].astype(
                    self.compute_dtype
                )
                mx.eval(logits, [x.state for x in cache])
            yield {
                "phase": "batched_decode",
                "tokens": index + 1,
                "batch_size": count,
                "generated_tokens": [len(tokens) for tokens in sampled],
                "finished_requests": [
                    ended or index + 1 == maximum for ended in finished
                ],
            }
        return [
            {
                "sampled_tokens": sampled[row],
                "behavior_logprobs": probabilities[row],
                "prompt_tokens": length,
                "completion_tokens": len(sampled[row]),
                "policy": body["policy"],
                "policy_revision": body["revision"],
                "finish_reason": "stop" if finished[row] else "length",
                "cache_precision": precision,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": time.monotonic() - decode_started,
                "batch_size": count,
                "execution_metrics": self.metrics(started),
            }
            for row, body in enumerate(bodies)
        ]

    def close_client(self, client):
        entry = self.clients.pop(client, None)
        self.reserved.pop(("optimizer", client), None)
        if entry and self.resident.selected is entry["adapter"].state():
            self.resident.select(self.idle_factors)
        while self._evict_policy():
            pass

    def close(self):
        self.clients.clear()
        self.reserved.clear()
        self.policies.clear()
        self.policy_sources.clear()
        self.policy_users.clear()
        self.idle_factors.clear()
        self.prefixes.clear()
        self.resident.frozen = []
        self.resident.selected = None
        self.resident.model = self.resident.lm = self.resident.adapter = self.lm = None
        import gc

        gc.collect()
        self.mx.clear_cache()


def _memory_level() -> int:
    """Return kern.memorystatus_level: the percent of host memory available."""
    return int(subprocess.check_output(
        ["sysctl", "-n", "kern.memorystatus_level"], text=True))
