"""Runtime source-only changes invalidate the actual engine/checkpoint identity."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from studio_compute import runtime
from studio_compute.engine import ResidentEngine
from studio_compute.storage import Store


@pytest.mark.parametrize("changed", ["runtime.py", "cache.py", "service.py", "cli.py", "storage.py", "protocol.py", "http.py", "activation_budget.py", "inference.py", "mlx_racing.py", "mlx_decode.py", "mlx_request_cache.py"])
def test_runtime_sources_cover_execution_and_durability_files(tmp_path, changed):
    originals = runtime.runtime_sources()
    for name in originals: (tmp_path / name).write_bytes((Path(runtime.__file__).parent / name).read_bytes())
    assert runtime.runtime_sources(tmp_path) == originals
    with (tmp_path / changed).open("ab") as file: file.write(b"\n# source-only change\n")
    altered = runtime.runtime_sources(tmp_path)
    assert {name for name in altered if altered[name] != originals[name]} == {changed}


def test_runtime_constructor_binds_hashes_to_checkpoint_identity(tmp_path, monkeypatch):
    def initialize(self, *a, **k):
        self.identity = {"model": "same-weights", "numerics": "same-math"}
        self.capabilities = {}
        self.resident = SimpleNamespace(budget=100)
        self.mx = SimpleNamespace()
    monkeypatch.setattr(ResidentEngine, "__init__", initialize)
    source = {"runtime.py": "old-runtime", "service.py": "same-service"}
    monkeypatch.setattr(runtime, "runtime_sources", lambda: dict(source))
    first = runtime.RuntimeEngine("unused")
    assert first.identity["runtime_source_sha256"] == first.capabilities["runtime"]["source_sha256"]
    store = Store(tmp_path)
    try:
        store.submit("save", "a", "create", {})
        state, _ = store.checkpoint("save", lambda folder: (folder / "factors").write_bytes(b"unchanged"),
                                    identity=first.identity, state={}, result={})
        source["runtime.py"] = "new-runtime"
        second = runtime.RuntimeEngine("unused")
        assert second.identity != first.identity
        assert second.identity["model"] == first.identity["model"]
        with pytest.raises(ValueError, match="compute identity differs"):
            store.verify(state["checkpoint"], second.identity)
    finally: store.close()
