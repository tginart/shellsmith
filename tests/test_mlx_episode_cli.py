"""Standalone MLX admission through the public CLI and real host/capture flow.

Readiness, model/session, HTTP serving and live episode execution are external
fixtures. Config validation, host arguments, harvesting and records are real.
"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shellsmith import cli
from shellsmith.episode import smoke
from shellsmith.episode.gm_process import python_runtime
import shellsmith.mlx
import shellsmith.mlx.checkpoint
import shellsmith.mlx.identity
from shellsmith.policy import server
from shellsmith.train import preflight
from shellsmith.train.backend import TokenPrompt
from test_acceptance_audit import _rollout
from test_gm_python_runtime import runtime
from test_harvest import _record
from test_player_credentials import external_tinker_sdk


@pytest.fixture
def mlx_episode(tmp_path, monkeypatch, runtime):
    runtime.validate()
    effects = []
    identity = {"identity_sha256": "fixture-native-model"}
    session = SimpleNamespace(sampler=object(), policy_identity=identity,
                              policy_ref="mlx-base:" + identity["identity_sha256"])
    fixture = SimpleNamespace(effects=effects, ready=True, config=None, smoke_args=None, session=session)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(python_runtime, "current_python_runtime", lambda: runtime)

    def probe(document):
        effects.append("readiness")
        assert document["compute"]["backend"] == "mlx"
        return {"ready": fixture.ready, "blockers": [] if fixture.ready else ["Authorize fixture Codex"]}

    async def create_session(config, *, checkpoint_root):
        effects.append("model")
        assert effects[0] == "readiness"
        fixture.config = config
        return session

    def load_adapter(path):
        effects.append("load-adapter")
        session.policy_ref = str(path.resolve())

    def verify_adapter(path, *, kind, expected_identity):
        effects.append("verify-adapter")
        assert kind == "sampler" and expected_identity == identity

    session.load_adapter = load_adapter
    session.close = lambda: effects.append("session-closed")
    monkeypatch.setattr(preflight, "probe_document", probe)
    monkeypatch.setattr(shellsmith.mlx, "create_session", create_session)
    monkeypatch.setattr(shellsmith.mlx.identity, "resolve_identity", lambda settings: identity)
    monkeypatch.setattr(shellsmith.mlx.checkpoint, "verify_checkpoint", verify_adapter)

    class Endpoint:
        api_key = "fixture-private-player-key"

        def __init__(self, *, sampler, records_dir):
            assert sampler is session.sampler

        def __enter__(self):
            effects.append("endpoint-started")
            return self

        def __exit__(self, *args):
            effects.append("endpoint-stopped")

        def base_url(self, episode_id):
            return f"http://127.0.0.1:9/e/{episode_id}/v1"

        def take_records(self, episode_id):
            return [replace(_record("native-request"), prompt=TokenPrompt((1, 2, 3)), backend="mlx", session_id=episode_id,
                            policy_ref=session.policy_ref, policy_identity=identity)]

    monkeypatch.setattr(server, "PlayerEndpoint", Endpoint)

    async def execute(args, *, player_api_key):
        effects.append("episode")
        fixture.smoke_args = args
        assert args.gm_python_runtime is runtime
        assert args.player_provider == "shellsmith-mlx" and player_api_key == Endpoint.api_key
        root = _rollout(tmp_path / "executed-fixture")
        root.rename(args.output_dir)
        return args.output_dir, {"final_reward": 0.4, "trainable": True}

    monkeypatch.setattr(smoke, "run_live_smoke", execute)
    return fixture


def _config(tmp_path, train_settings):
    path = tmp_path / "episode.toml"
    path.write_text(f'[compute]\nbackend="mlx"\ngpu_lease_root="{tmp_path / "lease"}"\n'
                    f'[train]\n{train_settings}\n')
    return path


@pytest.mark.parametrize("train_settings", ["steps=0", "kl_penalty_coef=0.5\nkl_discount_factor=0.5"],
                         ids=["no-training-steps", "unused-kl"])
@pytest.mark.parametrize("adapter_selected", [False, True], ids=["base", "adapter"])
def test_mlx_episode_accepts_inference_settings_and_preserves_adapter_flow(
    tmp_path, mlx_episode, train_settings, adapter_selected,
):
    config = _config(tmp_path, train_settings)
    output = tmp_path / "episode-output"
    model_path = tmp_path / "local-model"
    adapter_path = tmp_path / "selected-adapter"
    argv = ["episode", "--player", "mlx", "--config", str(config), "--log-dir", str(output),
            "--mlx-model-path", str(model_path)]
    if adapter_selected:
        argv += ["--adapter-path", str(adapter_path)]
    assert cli.main(argv) == 0
    assert mlx_episode.effects == ["readiness", *(["verify-adapter"] if adapter_selected else []), "model",
                                   *(["load-adapter"] if adapter_selected else []), "endpoint-started", "episode",
                                   "endpoint-stopped", "session-closed"]
    assert mlx_episode.config.mlx.model_path == str(model_path.resolve())
    args = mlx_episode.smoke_args
    assert args.player_context_window == 32768 and args.player_max_tokens == 8195
    assert args.player_thinking == "medium"
    assert args.player_renderer == "qwen3_8_medium_reasoning"
    assert (args.player_compaction_reserve_tokens, args.player_compaction_keep_recent_tokens) == (12288, 8192)
    records = json.loads((output / "player_records.json").read_text())
    assert len(records) == 1 and records[0]["backend"] == "mlx"
    assert records[0]["policy_ref"] == (str(adapter_path.resolve()) if adapter_selected else mlx_episode.session.policy_ref)
    assert records[0]["prompt_token_ids"] == [1, 2, 3] and records[0]["sampled_tokens"] == [10, 20]
    capture = json.loads((output / "capture_summary.json").read_text())
    assert capture["accepted_for_training"] is True and capture["captured_turns"] == 1
    assert capture["policy_identity"] == mlx_episode.session.policy_identity
    assert capture["policy_ref"] == records[0]["policy_ref"]
    assert capture["policy_kind"] == ("sampler_checkpoint" if adapter_selected else "base")


def test_mlx_episode_still_rejects_unsupported_profile_before_readiness(tmp_path, mlx_episode):
    config = _config(tmp_path, "steps=0")
    config.write_text(config.read_text() + '[mlx]\ncontext_window=16384\n')
    output = tmp_path / "episode-output"
    with pytest.raises(ValueError, match="context_window must be 32768"):
        cli.main(["episode", "--player", "mlx", "--config", str(config), "--log-dir", str(output)])
    assert not mlx_episode.effects and not output.exists() and not (tmp_path / "lease").exists()


def test_mlx_episode_login_failure_still_precedes_model_and_output(tmp_path, mlx_episode):
    config = _config(tmp_path, "steps=0")
    mlx_episode.ready = False
    output = tmp_path / "episode-output"
    with pytest.raises(SystemExit, match="Authorize fixture Codex"):
        cli.main(["episode", "--player", "mlx", "--config", str(config), "--log-dir", str(output)])
    assert mlx_episode.effects == ["readiness"]
    assert not output.exists() and not (tmp_path / "lease").exists()


@pytest.mark.parametrize("player_flags", [[], ["--player", "openai"], ["--player", "tinker"]],
                         ids=["default-openai", "explicit-openai", "tinker"])
@pytest.mark.parametrize("host_flags", [["--host"], ["--docker", "--image", "shellsmith:fixture"]],
                         ids=["host", "docker"])
@pytest.mark.parametrize("option", ["--config", "--mlx-model-path", "--adapter-path"])
def test_non_mlx_episode_rejects_mlx_options_before_credentials_or_execution(
    tmp_path, monkeypatch, player_flags, host_flags, option, capsys,
):
    import shellsmith.policy.deps
    import shellsmith.sandbox.docker_launcher
    effects = []

    def forbidden(boundary):
        def execute(*args, **kwargs):
            effects.append(boundary)
            pytest.fail(f"Unexpected {boundary} before {option} rejection")
        return execute

    monkeypatch.setattr(shellsmith.policy.deps, "load_tinker_deps", forbidden("Tinker credentials/model"))
    monkeypatch.setattr(smoke, "run_live_smoke", forbidden("host episode"))
    monkeypatch.setattr(shellsmith.sandbox.docker_launcher, "run_episode_in_docker", forbidden("Docker episode"))
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit, match=f"{option} requires --player mlx"):
        cli.main(["episode", *player_flags, *host_flags, option, str(tmp_path / "mlx-option-value"),
                  "--log-dir", str(output)])
    assert effects == [] and not output.exists() and not capsys.readouterr().out


def test_tinker_episode_without_mlx_options_keeps_default_model_and_capture(
    tmp_path, monkeypatch, external_tinker_sdk,
):
    effects = []
    output = tmp_path / "tinker-output"

    class Endpoint:
        api_key = "fixture-private-player-key"

        def __init__(self, *, sampler, records_dir, host):
            assert sampler.config.base_model == cli.DEFAULT_TINKER_BASE_MODEL
            assert sampler.config.thinking_renderer == cli.DEFAULT_PLAYER_THINKING_RENDERER
            assert host == "127.0.0.1"

        def __enter__(self):
            effects.append("endpoint-started")
            return self

        def __exit__(self, *args):
            effects.append("endpoint-stopped")

        def base_url(self, episode_id, *, host):
            assert host is None
            return f"http://127.0.0.1:9/e/{episode_id}/v1"

        def take_records(self, episode_id):
            return [replace(_record("tinker-request"), prompt=TokenPrompt((1, 2, 3)), session_id=episode_id)]

    async def execute(args, *, player_api_key):
        effects.append("episode")
        assert args.player_provider == "shellsmith-tinker" and args.player_model == server.SERVED_MODEL_ID
        assert args.player_context_window == cli.TINKER_CONTEXT_WINDOW_TOKENS
        assert player_api_key == Endpoint.api_key
        root = _rollout(tmp_path / "executed-tinker")
        root.rename(args.output_dir)
        return args.output_dir, {"final_reward": 0.4, "trainable": True}

    monkeypatch.setattr(server, "PlayerEndpoint", Endpoint)
    monkeypatch.setattr(smoke, "run_live_smoke", execute)
    assert cli.main(["episode", "--player", "tinker", "--log-dir", str(output)]) == 0
    assert external_tinker_sdk.calls == ["service", "capabilities", "sampling-client"]
    assert effects == ["endpoint-started", "episode", "endpoint-stopped"]
    records = json.loads((output / "player_records.json").read_text())
    assert records[0]["backend"] == "tinker" and records[0]["prompt_token_ids"] == [1, 2, 3]
    capture = json.loads((output / "capture_summary.json").read_text())
    assert capture["accepted_for_training"] is True and capture["captured_turns"] == 1


@pytest.mark.parametrize("values", [
    {"player_max_retries": 0, "gm_artifact_attempts": 1, "player_enabled": False},
    {"player_max_retries": 5, "gm_artifact_attempts": 2, "player_base_delay_ms": 17,
     "provider_max_retry_delay_ms": 77, "thinking_anomaly_attempts": 1},
])
def test_standalone_mlx_episode_forwards_all_configured_retries(tmp_path, mlx_episode, values):
    from dataclasses import asdict
    path = _config(tmp_path, "steps=0")
    source = "[retry]\n" + "".join(f"{key} = {str(value).lower()}\n" for key, value in values.items())
    path.write_text(path.read_text() + source)
    assert cli.main(["episode", "--player", "mlx", "--config", str(path),
                     "--log-dir", str(tmp_path / "episode-output")]) == 0
    assert mlx_episode.smoke_args.protocol_retry == asdict(mlx_episode.config.retry)
    for key, value in values.items():
        assert mlx_episode.smoke_args.protocol_retry[key] == value
