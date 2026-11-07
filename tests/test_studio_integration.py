"""Coordinator integration, exact captures, publication ordering and CPU-only use."""

import asyncio
from dataclasses import replace
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from shellsmith.train.backend import player_profile
from shellsmith.train.config import (
    TrainConfig,
    RunSection,
    ComputeSection,
    validate_train_config,
)
from shellsmith.train.studio import StudioPlayerSampler, RemoteSamplingSession
from shellsmith.train.trainer import _recover_studio_pending, latest_full_checkpoint
from shellsmith.train.durable_metrics import MetricsJournal


def test_template_dependencies_checked_before_rollouts(monkeypatch):
    from shellsmith.train import studio

    client = SimpleNamespace(tokenizer_bundle=lambda: {"hash": "bundle"})
    monkeypatch.setattr(studio, "materialize", lambda bundle: "cpu-assets")

    class Renderer:
        def __init__(self, path, *, reasoning_effort):
            assert path == "cpu-assets"
            assert reasoning_effort == "medium"

        def render(self, payload, *, thinking):
            assert payload["tools"] and payload["messages"]
            raise ImportError("chat template dependency missing")

    monkeypatch.setattr(studio, "Renderer", Renderer)
    with pytest.raises(ImportError, match="chat template dependency missing"):
        studio.load_renderer(client, {"tokenizer_hash": "bundle"})

    modes = []
    Renderer.render = lambda self, payload, *, thinking: modes.append(thinking)
    studio.load_renderer(client, {"tokenizer_hash": "bundle"})
    assert modes == [True, False]
    with pytest.raises(ValueError, match="Tokenizer identity changed"):
        studio.load_renderer(client, {"tokenizer_hash": "changed"})


def test_remote_configuration_no_model_paths_or_tinker_imports(tmp_path):
    config = TrainConfig(
        run=RunSection(run_id="studio-test"), compute=ComputeSection(backend="studio")
    )
    validate_train_config(config)
    profile = player_profile(config)
    assert profile["context_window"] == 262144 and profile["backend"] == "studio"
    assert "studio" in config.to_jsonable()
    assert "studio" not in TrainConfig().to_jsonable()
    assert "model_path" not in profile
    assert config.gm.max_world_steps == 256
    assert config.studio.thinking_tokens == 0
    assert profile["resolved_reasoning_tokens"] == 4096
    assert config.studio.response_tokens == 32768
    assert config.studio.reasoning_effort == "medium"
    assert profile["max_tokens"] == 36867
    assert profile["thinking_renderer"] == "qwen3_8_medium_reasoning"
    source = "import sys; import shellsmith.train.studio; assert not any(x=='mlx' or x.startswith('mlx.') or x=='tinker' for x in sys.modules)"
    subprocess.run([sys.executable, "-c", source], check=True)
    with pytest.raises(ValueError, match="KL disabled"):
        validate_train_config(
            replace(config, train=replace(config.train, kl_penalty_coef=0.01))
        )


def test_studio_phase_limits_are_independent_and_effort_is_part_of_profile_identity():
    from shellsmith.train.backend import profile_hash

    config = TrainConfig(run=RunSection(run_id="limits"), compute=ComputeSection(backend="studio"))
    changed = replace(config, studio=replace(config.studio, thinking_tokens=1024, response_tokens=6000))
    validate_train_config(changed)
    assert player_profile(changed)["max_tokens"] == 7027
    assert profile_hash(player_profile(config)) != profile_hash(player_profile(changed))
    high = replace(config, studio=replace(config.studio, reasoning_effort="xhigh"))
    assert profile_hash(player_profile(config)) != profile_hash(player_profile(high))
    with pytest.raises(ValueError, match="reasoning_effort"):
        validate_train_config(replace(config, studio=replace(config.studio, reasoning_effort="high")))


def test_frozen_coordinator_includes_and_verifies_studio_client(tmp_path):
    from shellsmith.jobs.bundle import pack, unpack, verify

    source = tmp_path / "source"
    for name, contents in {
        "src/shellsmith/jobs/worker.py": "# coordinator\n",
        "src/studio_compute/client.py": "# portable RPC\n",
        "src/studio_compute/numerics/MLX_VLM_LICENSE.txt": "license notice\n",
    }.items():
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(contents)
    bundle = pack(source)
    frozen = unpack(tmp_path / "runtime", bundle)
    assert (frozen / "src/studio_compute/client.py").read_text() == "# portable RPC\n"
    assert (frozen / "src/studio_compute/numerics/MLX_VLM_LICENSE.txt").is_file()
    (frozen / "src/studio_compute/client.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        verify(frozen, bundle["source_sha256"])


def test_rollout_policy_pinned_across_publication():
    backend = SimpleNamespace(
        policy_ref="p0",
        renderer=None,
        policy_identity={},
        settings=SimpleNamespace(model_id="27b", max_tokens=0, thinking_tokens=4096,
                                 response_tokens=4096, compaction_tokens=2048, plain_tokens=0),
    )
    sampler = StudioPlayerSampler(backend)
    assert sampler.pin("episode-a") == "p0"
    backend.policy_ref = "p1"
    assert sampler.pin("episode-a") == "p0"
    assert sampler.pin("episode-b") == "p1"
    sampler.release("episode-a")
    assert sampler.pin("episode-a") == "p1"


def test_exact_rpc_cursor_preserves_forced_ids():
    calls = []

    class RPC:
        def call(self, kind, body, *, request_id):
            calls.append((kind, body, request_id))
            return {
                "sampled_tokens": [7, 8],
                "behavior_logprobs": [-1.0, -2.0],
                "finish_reason": "length",
                "logprobs": [-3.0],
            }

    backend = SimpleNamespace(
        client=RPC(),
        renderer=None,
        policy_identity={},
        settings=SimpleNamespace(cache_precision="bf16"),
    )
    backend.sampling_request_id = lambda request: "fixture-client:" + request
    session = RemoteSamplingSession(backend, "p0", "request")
    ids, lp, reason, cursor = session.draw_pass([1, 2, 3], maximum=2, stop_ids={9})
    forced, cursor = session.force_tokens(cursor, [4])
    session.draw_pass(cursor, maximum=2, stop_ids={9})
    assert calls[-1][1]["prompt_token_ids"] == [1, 2, 3, 7, 8, 4]
    assert all(body["policy"] == "p0" for _, body, _ in calls)
    assert len({request for _, _, request in calls}) == 3


def test_committed_service_receipt_repairs_cursor_before_publication(tmp_path):
    path = tmp_path / "state.json"
    state = {
        "run_id": "r",
        "step": 0,
        "rft_updates_completed": 0,
        "next_world_offset": 0,
        "settled_world_offsets": [],
        "segments": [{}],
        "committed_group_ids": [],
        "checkpoints": [],
        "compute_backend": "studio",
        "policy_identity": {"model": "m"},
        "optimizer_update_in_flight": True,
        "pending_studio_update": {
            "id": "u",
            "step": 0,
            "started": 0.0,
            "metrics": {
                "step": 0,
                "datums": 2,
                "next_world_offset": 2,
                "settled_world_offsets": [],
            },
            "group_ids": ["g"],
            "next_world_offset": 2,
            "settled_world_offsets": [],
        },
    }
    journal = MetricsJournal.recover(
        tmp_path / "metrics.jsonl", state=state, checkpoint=None
    )

    class Backend:
        async def operation_status(self, request_id):
            if request_id.endswith("optimizer"):
                return {
                    "status": "succeeded",
                    "result": {
                        "checkpoint": "studio-state:committed",
                        "optim_metrics": {"optimizer_steps": 1},
                        "execution_metrics": {"seconds": 2.0},
                    },
                }
            return {"status": "succeeded", "result": {"forward_metrics": {"loss": 3.0},
                                                      "execution_metrics": {"seconds": 9.0}}}

        async def restore_state(self, reference, *, request_id):
            assert reference == "studio-state:committed"

        async def save_checkpoint(self, *, kind, name, metadata=None):
            return "studio-policy:p1" if kind == "sampler" else "studio-state:c1"

        async def publish_policy(self, *, revision):
            durable = json.loads(path.read_text())
            assert durable["step"] == 1 and durable["next_world_offset"] == 2
            assert (
                durable["committed_group_ids"] == ["g"]
                and "pending_studio_update" not in durable
            )
            assert (tmp_path / "metrics.jsonl").read_text()
            return "studio-policy:p1"

    asyncio.run(
        _recover_studio_pending(
            backend=Backend(), state=state, state_path=path, journal=journal
        )
    )
    assert latest_full_checkpoint(state)["path"] == "studio-state:c1"
    assert latest_full_checkpoint(state)["backend"] == "studio"
    recovered = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert recovered["backward_execution_metrics"]["seconds"] == 9.0
    assert recovered["optimizer_execution_metrics"]["seconds"] == 2.0
    assert recovered["optimizer_updates_committed"] == 1
    receipts = [json.loads(path.read_text())
                for path in (tmp_path / "policy-publications").glob("*.json")]
    assert len(receipts) == 1
    assert receipts[0]["policy_ref"] == "studio-policy:p1"
    assert receipts[0]["context"] == "recovery" and receipts[0]["seconds"] >= 0


def test_uncommitted_receipt_keeps_previous_restored_cursor(tmp_path):
    state = {
        "step": 2,
        "next_world_offset": 4,
        "optimizer_update_in_flight": False,
        "pending_studio_update": {"id": "u"},
    }

    class Backend:
        async def operation_status(self, request_id):
            return {"status": "interrupted"}

    asyncio.run(
        _recover_studio_pending(
            backend=Backend(),
            state=state,
            state_path=tmp_path / "state.json",
            journal=None,
        )
    )
    assert (
        state["step"] == 2
        and state["next_world_offset"] == 4
        and "pending_studio_update" not in state
    )


@pytest.mark.parametrize("model", ["typo-27B", "Qwen/another-model"])
def test_model_mismatch_rejected_before_tokenizer_or_client_creation(monkeypatch, model):
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="model-check"), compute=ComputeSection(backend="studio"))
    rpc = SimpleNamespace(capabilities=lambda: {"model": model, "context_window": 262144})
    monkeypatch.setattr(studio, "Client", lambda *a, **k: rpc)
    monkeypatch.setattr(studio, "load_renderer", lambda *a, **k: pytest.fail("must reject before tokenizer work"))
    with pytest.raises(ValueError, match="Configured model differs"):
        studio.StudioBackend(config)


@pytest.mark.parametrize("coefficient", [0.01, -1.0])
def test_studio_inference_ignores_unused_kl_training_settings(coefficient):
    config = TrainConfig(run=RunSection(run_id="inference-kl"), compute=ComputeSection(backend="studio"))
    config = replace(config, train=replace(config.train, kl_penalty_coef=coefficient))
    validate_train_config(config, campaign_mode="inference")
    with pytest.raises(ValueError, match="KL disabled|non-negative"):
        validate_train_config(config)


@pytest.mark.parametrize("window,model,blocker", [
    (131072, "Qwen/Qwen3.8-27B", None),
    (262144, "Qwen/Qwen3.8-27B", None),
    (262145, "Qwen/Qwen3.8-27B", "context exceeds"),
    (262144, "wrong-model", "model differs"),
])
@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_studio_readiness_matches_backend_context_and_model_rules(monkeypatch, window, model, blocker, effort):
    from shellsmith import cli
    from shellsmith.train import harness_identity, studio, world_selection
    import studio_compute.client
    config = TrainConfig(compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, context_window=window, reasoning_effort=effort))
    monkeypatch.setattr(world_selection, "select_world_stream", lambda config: SimpleNamespace(worlds=[]))
    monkeypatch.setattr(harness_identity, "codex_identity", lambda: {"executable_sha256": "fixture"})
    monkeypatch.setattr(harness_identity, "pi_sdk_identity", lambda *a: {"module_sha256": "fixture"})
    rpc = SimpleNamespace(capabilities=lambda: {"model": model, "context_window": 262144, "cache_precisions": ["bf16"]})
    monkeypatch.setattr(studio_compute.client, "Client", lambda *a: rpc)
    rendered = []
    def load(*args, reasoning_effort):
        rendered.append(reasoning_effort)
    monkeypatch.setattr(studio, "load_renderer", load)
    report = cli._readiness(config)
    assert report["ready"] == (blocker is None)
    assert rendered == [effort]
    if blocker:
        assert any(blocker in item for item in report["blockers"])


@pytest.mark.parametrize("effort", ["low", "xhigh"])
def test_readiness_fails_when_configured_reasoning_template_fails(monkeypatch, effort):
    from shellsmith import cli
    from shellsmith.train import harness_identity, studio, world_selection
    import studio_compute.client
    config = TrainConfig(compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, reasoning_effort=effort))
    monkeypatch.setattr(world_selection, "select_world_stream", lambda config: SimpleNamespace(worlds=[]))
    monkeypatch.setattr(harness_identity, "codex_identity", lambda: {"executable_sha256": "fixture"})
    monkeypatch.setattr(harness_identity, "pi_sdk_identity", lambda *a: {"module_sha256": "fixture"})
    rpc = SimpleNamespace(capabilities=lambda: {"model": config.studio.model_id, "context_window": 262144, "cache_precisions": ["bf16"]})
    monkeypatch.setattr(studio_compute.client, "Client", lambda *a: rpc)
    def render(*args, reasoning_effort="medium"):
        if reasoning_effort == effort:
            raise ValueError("configured template branch unavailable")
    monkeypatch.setattr(studio, "load_renderer", render)
    report = cli._readiness(config)
    assert not report["ready"] and "chat_template_checked" not in report
    assert any("configured template branch unavailable" in item for item in report["blockers"])


@pytest.mark.parametrize("kind,expected", [
    ("cispo", {"loss": -2.5, "loss_kind": "cispo", "cispo_loss": -2.5}),
    ("weighted_cross_entropy", {"loss": -2.5, "loss_kind": "weighted_cross_entropy", "weighted_cross_entropy_loss": -2.5}),
])
def test_replay_metrics_identify_the_requested_objective(kind, expected):
    from studio_compute.engine import loss_metrics
    assert loss_metrics(kind, -2.5) == expected
