"""Optional account pin for host Codex GMs; the selected store owns refresh."""

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time


GUARD_ENV = "SHELLSMITH_GM_AUTH_GUARD"


def _claims(token):
    try:
        part = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (ValueError, IndexError, TypeError) as error:
        raise RuntimeError("Cannot establish the cached Codex account identity.") from error


def _fingerprint(subject, account):
    if not all(isinstance(value, str) and value for value in (subject, account)):
        raise RuntimeError("Cannot establish the cached Codex account identity.")
    return hashlib.sha256(json.dumps({"subject": subject, "account_id": account}, sort_keys=True).encode()).hexdigest()


def _registry_entry(source):
    entry = json.loads(Path(source).read_text())["oauth/openai-codex"]
    claims = _claims(entry.get("key", ""))
    account = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    if entry.get("auth_mode") != "open_ai_codex" or not account or account != entry.get("account_id"):
        raise RuntimeError("Bladerunner Codex account metadata does not match its access token.")
    return entry, claims


def registry_identity(home, source):
    """Identify the owner's registry without copying or rotating its credentials."""
    entry, claims = _registry_entry(source)
    email = (claims.get("https://api.openai.com/profile") or {}).get("email")
    if not isinstance(email, str) or not email or email != entry.get("email"):
        raise RuntimeError("Cannot establish the Bladerunner Codex account email.")
    return {"codex_home": str(Path(home).resolve()), "credential_store": "bladerunner_registry",
            "source": str(Path(source).resolve()), "email": email,
            "account_fingerprint": _fingerprint(claims.get("sub"), entry["account_id"])}


def _pinned_entry(expected):
    entry, claims = _registry_entry(expected["source"])
    # Check the exact snapshot whose token/header will be used, even if the
    # owning application changes accounts between earlier identity reads.
    if _fingerprint(claims.get("sub"), entry["account_id"]) != expected["account_fingerprint"]:
        raise RuntimeError("Codex GM account/home changed; refusing to launch this turn.")
    return entry, claims


def _file_entry(home):
    document = json.loads((Path(home) / "auth.json").read_text())
    tokens = document.get("tokens") or {}
    if document.get("auth_mode") != "chatgpt" or not all(
        tokens.get(name) for name in ("id_token", "access_token", "refresh_token", "account_id")
    ):
        raise RuntimeError("GM account pin requires file-backed ChatGPT login.")
    claims = _claims(tokens["id_token"])
    if not all(isinstance(claims.get(key), str) and claims[key] for key in ("sub", "email")):
        raise RuntimeError("Cannot establish the cached Codex account identity.")
    return tokens, claims


def file_identity(home):
    """Read stable identity claims only; the selected login remains refresh owner."""
    home = Path(home).expanduser().resolve()
    tokens, claims = _file_entry(home)
    return {"codex_home": str(home), "credential_store": "file", "email": claims["email"],
            "account_fingerprint": _fingerprint(claims["sub"], tokens["account_id"])}


def _pinned_file_entry(expected):
    # Verify the same immutable read whose bearer is returned to the child.
    tokens, claims = _file_entry(expected["codex_home"])
    access = _claims(tokens["access_token"])
    account = (access.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    if (_fingerprint(claims["sub"], tokens["account_id"]) != expected["account_fingerprint"]
            or access.get("sub") != claims["sub"] or account != tokens["account_id"]):
        raise RuntimeError("Codex GM account/home changed; refusing to launch this turn.")
    return {"account_id": account, "key": tokens["access_token"]}, access


def verify_guard(executable, *, environ=None):
    """Fail before each GM turn if its selected home, executable or account changes."""
    env = os.environ if environ is None else environ
    if not env.get(GUARD_ENV):
        return []
    expected = json.loads(Path(env[GUARD_ENV]).read_text())
    home = Path(env.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
    registry = expected.get("credential_store") == "bladerunner_registry"
    actual = registry_identity(home, expected["source"]) if registry else file_identity(home)
    if any(actual.get(key) != expected.get(key) for key in actual):
        raise RuntimeError("Codex GM account/home changed; refusing to launch this turn.")
    if str(Path(executable).resolve()) != expected.get("executable"):
        raise RuntimeError("Codex GM executable changed; refusing to launch this turn.")
    if expected.get("executable_sha256") and hashlib.sha256(Path(executable).resolve().read_bytes()).hexdigest() != expected["executable_sha256"]:
        raise RuntimeError("Codex GM executable changed; refusing to launch this turn.")
    entry, _ = _pinned_entry(expected) if registry else _pinned_file_entry(expected)
    # Command refresh verifies the exact fetched credential for both stores.
    # No bearer/refresh token enters argv, environment, copies, or audit logs.
    provider = ('{ name="Shellsmith pinned Codex OAuth", '
                'base_url="https://chatgpt.com/backend-api/codex", wire_api="responses", '
                'http_headers={ "ChatGPT-Account-Id"=' + json.dumps(entry["account_id"]) + ' }, '
                'auth={ command=' + json.dumps(sys.executable) + ', args=' + json.dumps([
                    str(Path(__file__).resolve()), "--access-token", str(Path(env[GUARD_ENV]).resolve())
                ]) + ', timeout_ms=5000, refresh_interval_ms=60000 } }')
    return ["-c", 'model_provider="shellsmith_codex_oauth"', "-c",
            "model_providers.shellsmith_codex_oauth=" + provider]


def command_token(guard_path):
    """Private stdout protocol for Codex's command-backed bearer provider.

    The selected login remains refresh owner. Each CLI credential fetch rereads
    its store and rejects an account change; no refresh token is used here.
    """
    expected = json.loads(Path(guard_path).read_text())
    verify_guard(expected["executable"], environ={**os.environ, GUARD_ENV: str(guard_path)})
    if expected.get("credential_store") == "bladerunner_registry":
        entry, claims = _pinned_entry(expected)
    elif expected.get("credential_store") == "file":
        entry, claims = _pinned_file_entry(expected)
    else:
        raise RuntimeError("Expected a supported Codex account pin.")
    if claims.get("exp", 0) <= time.time() + 30:
        raise RuntimeError("Pinned Codex access token expired; refresh the owning login.")
    return entry["key"]


if __name__ == "__main__":
    # This mode must only be invoked by the CLI's internal credential provider,
    # never as an interactive shell command whose stdout is logged.
    if len(sys.argv) != 3 or sys.argv[1] != "--access-token":
        raise SystemExit("Expected internal Codex credential-provider invocation.")
    try:
        token = command_token(sys.argv[2])
    except Exception:
        raise SystemExit("Pinned Codex credential unavailable; inspect the owning login.") from None
    sys.stdout.write(token + "\n")
