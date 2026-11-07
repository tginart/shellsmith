"""Terminal request compaction preserves hash deduplication and recovery links."""

import json

import pytest

from studio_compute.protocol import Conflict
from studio_compute.storage import Store


@pytest.mark.parametrize("status", ["succeeded", "failed", "interrupted"])
def test_terminal_large_request_is_compacted_without_losing_deduplication(tmp_path, status):
    store = Store(tmp_path)
    try:
        body = {"prompt_token_ids": [123] * 100000, "advantages": [0.5] * 100000}
        store.submit("large", "a", "forward_backward", body)
        original = store.db.execute("SELECT hash,length(body) FROM operations").fetchone()
        assert original[1] > 700000
        store.running("large", {"phase": "replay"})
        assert store.db.execute("SELECT length(body) FROM operations").fetchone()[0] == original[1]
        if status == "succeeded": store.finish("large", {"forward_metrics": {"loss": 1.0}, "logprobs": [-0.5]})
        else: store.fail("large", "injected", status)
        row = store.db.execute("SELECT hash,body,progress FROM operations").fetchone()
        assert row[0] == original[0] and row[1] == "{}" and row[2] is None
        op, fresh = store.submit("large", "a", "forward_backward", body)
        assert not fresh and op["status"] == status
        with pytest.raises(Conflict): store.submit("large", "a", "forward_backward", {"different": True})
        if status == "succeeded": assert op["result"]["logprobs"] == [-0.5]
    finally: store.close()


def test_optimizer_link_survives_compaction_and_fresh_process_recovery(tmp_path):
    store = Store(tmp_path)
    store.submit("back", "a", "forward_backward", {"large": [1] * 1000}); store.finish("back", {})
    store.submit("unused", "a", "forward_backward", {"large": [1] * 1000}); store.finish("unused", {})
    store.submit("optim", "a", "optim_step", {"learning_rate": 0.1, "backward_operation": "back"}); store.finish("optim", {})
    assert json.loads(store.db.execute("SELECT body FROM operations WHERE id='optim'").fetchone()[0]) == {"backward_operation": "back"}
    store.close(); store = Store(tmp_path)
    try:
        store.recover({})
        assert store.operation("back")["status"] == "succeeded"
        assert store.operation("unused")["status"] == "interrupted"
    finally: store.close()


def test_interrupted_create_rehydrates_and_old_terminal_rows_compact(tmp_path):
    store = Store(tmp_path)
    try:
        body = {"seed": 7, "learning_rate": 0.001}
        store.submit("create", "a", "create", body); store.fail("create", "interrupted", "interrupted")
        assert store.db.execute("SELECT body FROM operations").fetchone()[0] == "{}"
        op, fresh = store.submit("create", "a", "create", body)
        assert fresh and op["status"] == "queued"
        assert json.loads(store.db.execute("SELECT body FROM operations").fetchone()[0]) == body
        store.submit("old", "b", "sample", {"large": [1] * 10000}); store.finish("old", {"sampled_tokens": [2]})
        with store.db: store.db.execute("UPDATE operations SET body=? WHERE id='old'", (json.dumps({"old_payload": [1] * 10000}),))
        store.recover({})
        assert store.db.execute("SELECT body FROM operations WHERE id='old'").fetchone()[0] == "{}"
        assert store.operation("old")["result"] == {"sampled_tokens": [2]}
    finally: store.close()


def test_successive_large_requests_reuse_freed_sqlite_pages(tmp_path):
    store = Store(tmp_path)
    try:
        body = {"large": [12345] * 100000}
        counts = []
        for i in range(8):
            store.submit(str(i), "a", "forward_backward", body); store.finish(str(i), {})
            counts.append(store.db.execute("PRAGMA page_count").fetchone()[0])
        assert max(counts) <= 2 * counts[0]
        assert store.db.execute("SELECT SUM(length(body)) FROM operations").fetchone()[0] == 16
    finally: store.close()
