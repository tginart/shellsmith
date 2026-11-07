"""Transient token progress must not weaken durable operation boundaries."""

import pytest

from studio_compute.storage import Store


def test_live_token_progress_has_no_transactions_and_polls_latest_counts(tmp_path):
    store = Store(tmp_path)
    try:
        for index in range(8):
            name = str(index)
            store.submit(name, "a", "sample", {"seed": index})
            store.running(name)
        statements = []
        store.db.set_trace_callback(statements.append)
        for token in range(128):
            for index in range(8):
                store.observe_progress(
                    str(index), {"phase": "decode", "tokens": token + 1}
                )
        assert statements == []
        for index in range(8):
            row = store.operation(str(index))
            assert row["status"] == "running"
            assert row["progress"] == {"phase": "decode", "tokens": 128}
        assert (
            store.db.execute(
                "SELECT count(*) FROM operations WHERE progress IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
        store.finish("0", {"sampled_tokens": [3]})
        assert "0" not in store.live_progress
        assert store.operation("0")["progress"] is None
        assert statements.count("COMMIT") == 1
        store.running("0", {"tokens": 999})
        assert "0" not in store.live_progress
        store.fail("1", "canceled", "interrupted")
        assert "1" not in store.live_progress
    finally:
        store.close()


def test_restart_drops_only_progress_and_preserves_deduplication(tmp_path):
    body = {"seed": 7}
    store = Store(tmp_path)
    store.submit("live", "a", "sample", body)
    store.running("live")
    store.observe_progress("live", {"tokens": 129})
    store.submit("done", "a", "sample", body)
    store.running("done")
    store.observe_progress("done", {"tokens": 3})
    store.finish("done", {"sampled_tokens": [1, 2, 3]})
    store.close()
    store = Store(tmp_path)
    try:
        store.recover({})
        assert store.live_progress == {}
        assert store.operation("live")["status"] == "interrupted"
        assert store.operation("live")["progress"] is None
        receipt, fresh = store.submit("done", "a", "sample", body)
        assert not fresh
        assert receipt["status"] == "succeeded"
        assert receipt["result"]["sampled_tokens"] == [1, 2, 3]
        receipt, fresh = store.submit("live", "a", "sample", body)
        assert not fresh and receipt["status"] == "interrupted"
        with pytest.raises(KeyError, match="durable admission"):
            store.observe_progress("missing", {})
    finally:
        store.close()
