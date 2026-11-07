"""Keep tests independent of the checkout's live Backprop deployment."""

import pytest


@pytest.fixture(autouse=True)
def isolate_local_telemetry_connection(monkeypatch):
    """Explicit connection tests can still select their own temporary file."""
    monkeypatch.setattr("shellsmith.train.telemetry.LOCAL_CONNECTION_FILE", None)
