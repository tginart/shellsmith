"""Real HTTP handlers race episode closure at deterministic sampling boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from shellsmith.policy.server import PlayerEndpoint
from test_player_endpoint import _request
from test_sampling_budget import _sampler


@pytest.mark.parametrize("stage,requests", [("before_check", 1), ("before_pin", 1), ("provider", 1), ("provider", 2)])
def test_closed_episode_fences_handlers_and_releases_late_pins(tmp_path, monkeypatch, stage, requests):
    sampler, old_client, _ = _sampler(configured=16, thinking=False)
    _, new_client, _ = _sampler(configured=16, thinking=False)
    entered = threading.Barrier(requests + 1)
    proceed, finished = threading.Event(), threading.Event()
    endpoint = PlayerEndpoint(sampler=sampler, records_dir=tmp_path / "records")
    original_end = endpoint._end_episode_request
    def end(episode_id, state):
        original_end(episode_id, state)
        if episode_id != "cancelled" or "cancelled" not in endpoint._episodes:
            finished.set()
    monkeypatch.setattr(endpoint, "_end_episode_request", end)
    if stage == "before_check":
        original = endpoint._request_open
        def gate(state):
            if not proceed.is_set():
                entered.wait(timeout=5)
                assert proceed.wait(5)
            return original(state)
        monkeypatch.setattr(endpoint, "_request_open", gate)
    elif stage == "before_pin":
        original = sampler.sample
        def gate(*args, **kwargs):
            if not proceed.is_set():
                entered.wait(timeout=5)
                assert proceed.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(sampler, "sample", gate)
    else:
        original = old_client.sample
        def gate(**kwargs):
            if not proceed.is_set():
                entered.wait(timeout=5)
                assert proceed.wait(5)
            return original(**kwargs)
        monkeypatch.setattr(old_client, "sample", gate)
    payload = {"messages": [{"role": "user", "content": "continue"}], "reasoning_effort": "none"}
    endpoint.start()
    try:
        endpoint.open_episode("cancelled")
        sampler.pin("cancelled")
        sampler.set_sampling_client(new_client)
        with ThreadPoolExecutor(max_workers=requests) as pool:
            futures = [pool.submit(_request, endpoint.base_url("cancelled") + "/chat/completions",
                                   key=endpoint.api_key, body=payload) for _ in range(requests)]
            try:
                entered.wait(timeout=5)
                endpoint.close_episode("cancelled")
                assert endpoint._episodes["cancelled"].closed
                assert endpoint._episodes["cancelled"].active == requests
                assert endpoint._records == {}
            finally:
                proceed.set()
            assert all(future.result(timeout=10)[0] == 410 for future in futures)
        assert finished.wait(5)
        assert endpoint._episodes == {} and endpoint._records == {}
        assert sampler._session_clients == {}
        assert len(old_client.calls) == (0 if stage == "before_check" else requests)
        assert new_client.calls == []  # An admitted handler never switches policies.
        evidence = endpoint.records_dir / "cancelled.jsonl"
        if stage == "before_check":
            assert not evidence.exists()
        else:
            rows = [json.loads(line) for line in evidence.read_text().splitlines()]
            assert len(rows) == requests
            assert all(row["discarded_after_episode_close"] is True for row in rows)
        # New requests to the closed ID remain rejected without tombstone growth.
        for _ in range(3):
            assert _request(endpoint.base_url("cancelled") + "/chat/completions", key=endpoint.api_key, body=payload)[0] == 410
        assert endpoint._episodes == {} and sampler._session_clients == {}
        endpoint.open_episode("sibling")
        finished.clear()
        assert _request(endpoint.base_url("sibling") + "/chat/completions", key=endpoint.api_key, body=payload)[0] == 200
        assert finished.wait(5)
        assert len(endpoint.take_records("sibling")) == 1
        endpoint.close_episode("sibling")
        assert endpoint._episodes == {} and sampler._session_clients == {}
    finally:
        proceed.set()
        endpoint.stop()
