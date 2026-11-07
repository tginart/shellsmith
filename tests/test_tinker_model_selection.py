"""Real parser/setup checks; only the external Tinker SDK is a fixture.

No service authentication, sampling, or tokenizer download is needed. Allowing
an exact Qwen3.8 model name does not waive the service-reported context check.
"""

import builtins
from types import SimpleNamespace

import pytest

from shellsmith import cli
from shellsmith.config import (
    DEFAULT_PLAYER_PLAIN_RENDERER,
    DEFAULT_PLAYER_THINKING_RENDERER,
    DEFAULT_TINKER_BASE_MODEL,
    SUPPORTED_TINKER_BASE_MODELS,
)
from shellsmith.policy.sampler import CONTEXT_WINDOW_TOKENS
from shellsmith.policy.tinker_setup import base_sampling_client


UNSUPPORTED_MODELS = (
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.8-235B",
    "Qwen/Qwen3.8-27B:peft:65536",
    "Qwen/Qwen3.8-27B:peft:262144:extra",
    "qwen/qwen3.8-27b",
    "Qwen/Qwen3.8-27B ",
    "",
)


@pytest.fixture
def sdk():
    """External SDK boundary with observable initialization and call ordering."""
    state = SimpleNamespace(
        events=[],
        initialization_error=None,
        capabilities=SimpleNamespace(supported_models=[]),
        sampling_client=object(),
    )

    class ServiceClient:
        def __init__(self):
            state.events.append(("ServiceClient",))
            if state.initialization_error is not None:
                raise state.initialization_error

        def get_server_capabilities(self):
            state.events.append(("get_server_capabilities",))
            return state.capabilities

        def create_sampling_client(self, *, base_model):
            state.events.append(("create_sampling_client", base_model))
            return state.sampling_client

    state.deps = SimpleNamespace(tinker=SimpleNamespace(ServiceClient=ServiceClient))
    return state


@pytest.fixture
def forbid_optional_sdk_imports(monkeypatch):
    """An import boundary guard, not a replacement for Shellsmith's loader."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"tinker", "tinker_cookbook", "torch"}:
            pytest.fail(f"Optional SDK imported before model validation: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_allowlist_and_parser_default_keep_the_fixed_qwen38_contract():
    assert DEFAULT_TINKER_BASE_MODEL == "Qwen/Qwen3.8-27B:peft:262144"
    assert SUPPORTED_TINKER_BASE_MODELS == (
        DEFAULT_TINKER_BASE_MODEL,
        "Qwen/Qwen3.8-27B",
    )
    assert DEFAULT_PLAYER_THINKING_RENDERER == "qwen3_8_xhigh_reasoning"
    assert DEFAULT_PLAYER_PLAIN_RENDERER == "qwen3_8_disable_thinking"
    args = cli.build_parser().parse_args(["episode", "--player", "tinker"])
    assert args.tinker_model == DEFAULT_TINKER_BASE_MODEL


@pytest.mark.parametrize("model_name", UNSUPPORTED_MODELS)
def test_shipped_parser_rejects_unsupported_models(model_name, capsys, forbid_optional_sdk_imports):
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args(["episode", "--player", "tinker", "--tinker-model", model_name])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "--tinker-model: invalid choice" in error
    for allowed_name in SUPPORTED_TINKER_BASE_MODELS:
        assert allowed_name in error


@pytest.mark.parametrize("model_name", UNSUPPORTED_MODELS)
def test_direct_cli_call_rejects_before_optional_dependencies(model_name, tmp_path, forbid_optional_sdk_imports):
    args = cli.build_parser().parse_args(["episode", "--player", "tinker"])
    args.tinker_model = model_name  # Direct callers can bypass argparse choices.
    log_dir = tmp_path / "not-started"
    with pytest.raises(RuntimeError, match="Unsupported Tinker Player model") as caught:
        cli._run_with_tinker_player(args, log_dir)
    assert repr(model_name) in str(caught.value)
    assert not log_dir.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("model_name", [*UNSUPPORTED_MODELS, None])
def test_base_client_rejects_before_authenticated_service_initialization(model_name, sdk):
    sdk.initialization_error = RuntimeError("Authenticated SDK initialization must not run")
    with pytest.raises(ValueError, match="Unsupported Tinker base model") as caught:
        base_sampling_client(sdk.deps, model_name=model_name)
    assert repr(model_name) in str(caught.value)
    assert "Qwen3.8" in str(caught.value)
    assert sdk.events == []


@pytest.mark.parametrize("model_name", SUPPORTED_TINKER_BASE_MODELS)
@pytest.mark.parametrize("context_limit", [CONTEXT_WINDOW_TOKENS, 262_144])
def test_parser_accepted_names_are_checked_and_forwarded_exactly(model_name, context_limit, sdk):
    args = cli.build_parser().parse_args(["episode", "--player", "tinker", "--tinker-model", model_name])
    assert args.tinker_model == model_name
    other_name = next(name for name in SUPPORTED_TINKER_BASE_MODELS if name != model_name)
    sdk.capabilities.supported_models = [
        SimpleNamespace(model_name=other_name, max_context_length=65_536),
        SimpleNamespace(model_name=model_name, max_context_length=context_limit),
    ]
    assert base_sampling_client(sdk.deps, model_name=args.tinker_model) is sdk.sampling_client
    assert sdk.events == [
        ("ServiceClient",),
        ("get_server_capabilities",),
        ("create_sampling_client", model_name),
    ]


@pytest.mark.parametrize("model_name", SUPPORTED_TINKER_BASE_MODELS)
@pytest.mark.parametrize("limits, error", [
    pytest.param([65_536], "Shellsmith needs", id="short-context"),
    pytest.param([CONTEXT_WINDOW_TOKENS - 1], "Shellsmith needs", id="one-token-short"),
    pytest.param([], "exactly once", id="missing-model"),
    pytest.param([262_144, 262_144], "exactly once", id="duplicate-model"),
    pytest.param([None], "exactly once", id="missing-context"),
    pytest.param([True], "exactly once", id="boolean-context"),
    pytest.param(["262144"], "exactly once", id="string-context"),
    pytest.param([262_144.0], "exactly once", id="float-context"),
])
def test_allowed_names_still_require_unambiguous_sufficient_service_context(model_name, limits, error, sdk):
    other_name = next(name for name in SUPPORTED_TINKER_BASE_MODELS if name != model_name)
    sdk.capabilities.supported_models = [
        SimpleNamespace(model_name=other_name, max_context_length=262_144),
        *(SimpleNamespace(model_name=model_name, max_context_length=limit) for limit in limits),
    ]
    with pytest.raises(RuntimeError, match=error) as caught:
        base_sampling_client(sdk.deps, model_name=model_name)
    assert repr(model_name) in str(caught.value)
    if error == "Shellsmith needs":
        assert str(CONTEXT_WINDOW_TOKENS) in str(caught.value)
    assert sdk.events == [("ServiceClient",), ("get_server_capabilities",)]


@pytest.mark.parametrize("capabilities", [
    SimpleNamespace(),
    SimpleNamespace(supported_models=None),
    SimpleNamespace(supported_models=()),
])
def test_allowed_name_still_requires_a_service_model_list(capabilities, sdk):
    sdk.capabilities = capabilities
    with pytest.raises(RuntimeError, match="supported_models list"):
        base_sampling_client(sdk.deps, model_name=DEFAULT_TINKER_BASE_MODEL)
    assert sdk.events == [("ServiceClient",), ("get_server_capabilities",)]


@pytest.mark.parametrize("model_name", SUPPORTED_TINKER_BASE_MODELS)
def test_fixed_renderers_match_installed_cookbook_metadata_without_tokenizers(model_name):
    model_info = pytest.importorskip("tinker_cookbook.model_info", reason="optional installed Cookbook metadata check")
    recommended = model_info.get_recommended_renderer_names(model_name)
    assert recommended[0] == DEFAULT_PLAYER_THINKING_RENDERER
    assert DEFAULT_PLAYER_PLAIN_RENDERER in recommended
