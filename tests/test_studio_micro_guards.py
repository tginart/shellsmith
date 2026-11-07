"""Account rotation and workload caps fail before model/API work is launched."""

import asyncio
import base64
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shellsmith.episode.gm_process.auth_identity import GUARD_ENV, command_token, file_identity, registry_identity, verify_guard
from shellsmith.train.backend import TokenSequence
from shellsmith.train.config import ComputeSection, RunSection, TrainConfig
from shellsmith.train.trainer import submit_update


def login(home, *, subject="person-a", account="workspace-a", refresh="first"):
    claims = base64.urlsafe_b64encode(json.dumps({"sub": subject, "email": "test@example.com"}).encode()).decode().rstrip("=")
    access_claims = {"sub": subject, "exp": time.time() + 600, "nonce": refresh,
                     "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    access = "header." + base64.urlsafe_b64encode(json.dumps(access_claims).encode()).decode().rstrip("=") + ".signature"
    (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
        "id_token": "header." + claims + ".signature", "access_token": access,
        "refresh_token": refresh, "account_id": account}}))


def test_account_pin_allows_refresh_and_rejects_account_home_or_executable_change(tmp_path):
    login(tmp_path)
    binary = tmp_path / "codex"
    expected = {**file_identity(tmp_path), "executable": str(binary.resolve())}
    guard = tmp_path / "guard.json";guard.write_text(json.dumps(expected))
    env = {GUARD_ENV: str(guard), "CODEX_HOME": str(tmp_path)}
    flags = verify_guard(binary, environ=env)
    assert 'model_provider="shellsmith_codex_oauth"' in flags
    assert json.loads((tmp_path / "auth.json").read_text())["tokens"]["access_token"] not in str(flags)
    login(tmp_path, refresh="rotated")
    assert verify_guard(binary, environ=env)
    with pytest.raises(RuntimeError, match="executable changed"):
        verify_guard(tmp_path / "other", environ=env)
    login(tmp_path, account="workspace-b")
    with pytest.raises(RuntimeError, match="account/home changed"):
        verify_guard(binary, environ=env)
    login(tmp_path, subject="person-b")
    with pytest.raises(RuntimeError, match="account/home changed"):
        verify_guard(binary, environ=env)
    other = tmp_path / "other-home";other.mkdir();login(other)
    with pytest.raises(RuntimeError, match="account/home changed"):
        verify_guard(binary, environ={**env, "CODEX_HOME": str(other)})
    assert "rotated" not in json.dumps(expected)


def test_unpinned_existing_launches_are_unchanged():
    assert verify_guard(Path("codex"), environ={}) == []


@pytest.mark.parametrize("resume",[False,True])
def test_account_provider_overrides_follow_exec_resume_subcommand(tmp_path,monkeypatch,resume):
    from shellsmith.episode.gm_process import auth_identity
    from shellsmith.episode.gm_process.codex import CodexWorldConfig,CodexWorldProcess
    monkeypatch.setattr(auth_identity,"verify_guard",lambda executable:["-c",'model_provider="pinned"'])
    config=CodexWorldConfig(executable=tmp_path/"codex",model="gpt-6.1-sol",workspace_dir=tmp_path,
        log_dir=tmp_path/"logs",bootstrap_response_schema_path=tmp_path/"schema.json")
    command=CodexWorldProcess(config)._base_command(resume=resume)
    subcommand=command.index("resume") if resume else command.index("exec")
    assert command.index('model_provider="pinned"')>subcommand


def test_registry_provider_keeps_credentials_out_of_flags_and_follows_owner_refresh(tmp_path, monkeypatch):
    def registry(account="workspace-a", expiration=None):
        claims = {"sub":"person-a", "exp": expiration or time.time()+600,
                  "https://api.openai.com/auth":{"chatgpt_account_id":account},
                  "https://api.openai.com/profile":{"email":"test@example.com"}}
        payload=base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        token="header."+payload+".signature"
        source.write_text(json.dumps({"oauth/openai-codex":{"auth_mode":"open_ai_codex",
            "account_id":account,"email":"test@example.com","key":token,"refresh_token":"owner-only"}}))
        return token
    source=tmp_path/"registry.json";token=registry()
    expected={**registry_identity(tmp_path,source),"executable":str((tmp_path/"codex").resolve())}
    guard=tmp_path/"guard.json";guard.write_text(json.dumps(expected))
    monkeypatch.setenv("CODEX_HOME",str(tmp_path))
    env={GUARD_ENV:str(guard),"CODEX_HOME":str(tmp_path)}
    flags=verify_guard(expected["executable"],environ=env)
    assert 'model_provider="shellsmith_codex_oauth"' in flags
    assert token not in str(flags) and "owner-only" not in str(flags)
    assert not (tmp_path/"auth.json").exists()
    assert command_token(guard)==token
    refreshed=registry(expiration=time.time()+1200)
    assert command_token(guard)==refreshed
    registry(expiration=time.time()-60)
    with pytest.raises(RuntimeError,match="expired"):
        command_token(guard)
    registry(account="workspace-b")
    with pytest.raises(RuntimeError,match="account/home changed"):
        command_token(guard)
    registry()
    from shellsmith.episode.gm_process import auth_identity
    original=auth_identity._registry_entry
    calls=[]
    def switch_after_identity(source):
        value=original(source)
        calls.append(True)
        if len(calls)==1:registry(account="workspace-b")
        return value
    monkeypatch.setattr(auth_identity,"_registry_entry",switch_after_identity)
    with pytest.raises(RuntimeError,match="account/home changed"):
        command_token(guard)


@pytest.mark.parametrize("kind", ["prompt", "completion"])
def test_micro_budget_rejects_before_durable_intent_and_training_rpc(kind):
    sequence = TokenSequence((1, 2, 3), (4, 5), (-1., -1.), (.1, .1), (1., 1.), "sample")
    config = TrainConfig(compute=ComputeSection(backend="studio"))
    config = replace(config, train=replace(config.train, **{f"max_update_{kind}_tokens": 1}))
    backend = SimpleNamespace(update=AsyncMock())
    called = []
    with pytest.raises(ValueError, match=f"Update {kind} token budget exceeded"):
        asyncio.run(submit_update(deps=None, training_client=backend, reference_client=None,
            datums=[SimpleNamespace(datum=sequence)], config=config, before_submit=lambda: called.append(True)))
    backend.update.assert_not_called()
    assert not called


def test_exact_micro_budget_admits_whole_sequences():
    sequence = TokenSequence((1, 2, 3), (4, 5), (-1., -1.), (.1, .1), (1., 1.), "sample")
    config = TrainConfig(compute=ComputeSection(backend="studio"))
    config = replace(config, train=replace(config.train, max_update_prompt_tokens=3, max_update_completion_tokens=2))
    backend = SimpleNamespace(update=AsyncMock(return_value={"ok": True}))
    called = []
    assert asyncio.run(submit_update(deps=None, training_client=backend, reference_client=None,
        datums=[SimpleNamespace(datum=sequence)], config=config, before_submit=lambda: called.append(True))) == {"ok": True}
    assert called == [True]
    assert backend.update.call_args.args[0].sequences == (sequence,)


def test_smaller_client_context_uses_served_model_without_expanding_workload(monkeypatch):
    from shellsmith.train import studio
    config = TrainConfig(run=RunSection(run_id="context-check"), compute=ComputeSection(backend="studio"))
    config = replace(config, studio=replace(config.studio, context_window=8192))
    rpc = SimpleNamespace(capabilities=lambda:{"context_window":262144,"model":config.studio.model_id,"identity":{}, "cache_precisions": ["bf16"]})
    monkeypatch.setattr(studio,"Client",lambda *a,**k:rpc)
    monkeypatch.setattr(studio,"load_renderer",lambda *a, **k:None)
    backend = studio.StudioBackend(config)
    assert backend.settings.context_window == 8192
    config = replace(config, studio=replace(config.studio, context_window=262145))
    with pytest.raises(ValueError,match="exceeds"):
        studio.StudioBackend(config)


def test_file_command_provider_checks_each_fetched_snapshot_and_allows_same_account_refresh(tmp_path, monkeypatch):
    from shellsmith.episode.gm_process import auth_identity
    login(tmp_path)
    expected = {**file_identity(tmp_path), "executable": str((tmp_path / "codex").resolve())}
    guard = tmp_path / "guard.json"; guard.write_text(json.dumps(expected))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    flags = verify_guard(expected["executable"], environ={GUARD_ENV: str(guard), "CODEX_HOME": str(tmp_path)})
    before = (tmp_path / "auth.json").read_bytes()
    assert command_token(guard) == json.loads(before)["tokens"]["access_token"]
    assert (tmp_path / "auth.json").read_bytes() == before
    login(tmp_path, refresh="new-token")
    rotated = json.loads((tmp_path / "auth.json").read_text())["tokens"]["access_token"]
    assert command_token(guard) == rotated and rotated not in str(flags)
    # A login switch after turn launch must fail at credential fetch as well.
    login(tmp_path, account="different-workspace")
    with pytest.raises(RuntimeError, match="account/home changed"):
        command_token(guard)
    login(tmp_path)
    original = auth_identity._file_entry
    reads = []
    def switch_after_check(home):
        result = original(home)
        reads.append(True)
        if len(reads) == 1: login(tmp_path, subject="replacement-person")
        return result
    monkeypatch.setattr(auth_identity, "_file_entry", switch_after_check)
    with pytest.raises(RuntimeError, match="account/home changed"):
        command_token(guard)


def test_file_provider_rejects_foreign_bearer_under_unchanged_id_token(tmp_path, monkeypatch):
    login(tmp_path)
    expected = {**file_identity(tmp_path), "executable": str((tmp_path / "codex").resolve())}
    guard = tmp_path / "guard.json"; guard.write_text(json.dumps(expected))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    document = json.loads((tmp_path / "auth.json").read_text())
    login(tmp_path, account="replacement")
    document["tokens"]["access_token"] = json.loads((tmp_path / "auth.json").read_text())["tokens"]["access_token"]
    (tmp_path / "auth.json").write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="account/home changed"):
        command_token(guard)


def test_file_provider_command_rechecks_after_launch_without_disclosing_credentials(tmp_path, monkeypatch):
    import os
    import subprocess
    import tomllib
    login(tmp_path)
    expected = {**file_identity(tmp_path), "executable": str((tmp_path / "codex").resolve())}
    guard = tmp_path / "guard.json"; guard.write_text(json.dumps(expected))
    env = {**os.environ, GUARD_ENV: str(guard), "CODEX_HOME": str(tmp_path)}
    flags = verify_guard(expected["executable"], environ=env)
    provider = tomllib.loads(flags[3])["model_providers"]["shellsmith_codex_oauth"]
    command = [provider["auth"]["command"], *provider["auth"]["args"]]
    token = json.loads((tmp_path / "auth.json").read_text())["tokens"]["access_token"]
    fetched = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert fetched.returncode == 0 and fetched.stdout.strip() == token and fetched.stderr == ""
    login(tmp_path, account="other-account")
    other = json.loads((tmp_path / "auth.json").read_text())["tokens"]["access_token"]
    rejected = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert rejected.returncode != 0 and rejected.stdout == ""
    assert token not in rejected.stderr and other not in rejected.stderr
