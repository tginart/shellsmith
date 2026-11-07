import pytest

from shellsmith.campaign.evaluation import acceptance_interval, measurements, summarize_evaluation


def test_wilson_interval_includes_uncertainty_for_perfect_acceptance():
    result = acceptance_interval(16, 16)
    assert result["rate"] == 1
    assert result["ci95"] == pytest.approx([.8063923194655637, 1.])
    assert acceptance_interval(0, 0)["ci95"] is None


def test_reward_uses_accepted_population_and_usage_accounts_for_failures():
    rows = [
        {"accepted_count": 1, "audit": {"final_task": {"reward": .25}}, "wall_seconds": 3,
         "token_usage": {"prompt_tokens": 20, "completion_tokens": 4, "forced_close_tokens": 1}},
        {"accepted_count": 1, "audit": {"final_task": {"reward": .75}}, "wall_seconds": 5,
         "token_usage": {"prompt_tokens": 30, "completion_tokens": 8}},
        {"accepted_count": 0, "audit": {"final_task": {"reward": 1}}, "wall_seconds": 10,
         "token_usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ]
    result = summarize_evaluation(rows)
    assert result["accepted_reward"]["mean"] == .5
    assert result["accepted_reward"]["standard_error"] == pytest.approx(.25)
    assert result["acceptance"]["rate"] == pytest.approx(2 / 3)
    assert result["player_tokens"]["completion_tokens"] == 15
    assert result["player_tokens"]["forced_close_tokens"] == 1
    assert result["episode_latency_seconds"]["mean"] == 6


def test_missing_or_invalid_measurements_are_visible():
    assert measurements([True, float("nan"), None])["n"] == 0
    result = summarize_evaluation([{"accepted_count": 0}, {"accepted_count": 1}])
    assert result["accepted_reward"]["n"] == 0
    assert result["missing_latency_measurements"] == 2
    assert result["episodes_with_token_measurements"] == 0


def test_discarded_groups_count_costs_but_cannot_improve_evaluation_reward():
    eligible = [{"accepted_count": 1, "audit": {"final_task": {"reward": .25}}}]
    discarded = [{"accepted_count": 1, "audit": {"final_task": {"reward": 1.}},
                  "token_usage": {"completion_tokens": 10}, "wall_seconds": 8}]
    result = summarize_evaluation(eligible, discarded_outcomes=discarded)
    assert result["acceptance"]["rate"] == .5
    assert result["accepted_reward"]["n"] == 1
    assert result["accepted_reward"]["mean"] == .25
    assert result["player_tokens"]["completion_tokens"] == 10
    assert result["discarded_attempts"] == 1


def test_unlaunched_slots_do_not_count_as_attempts_or_measurements():
    actual = {"accepted_count": 1, "audit": {"final_task": {"reward": .25}},
              "token_usage": {"completion_tokens": 10}, "wall_seconds": 8}
    placeholder = {"episode_not_started": True, "accepted_count": 0}
    result = summarize_evaluation([actual, placeholder], discarded_outcomes=[placeholder])
    assert result["acceptance"]["accepted"] == result["acceptance"]["attempted"] == 1
    assert result["acceptance"]["rate"] == 1.
    assert result["not_started_attempt_slots"] == 2
    assert result["discarded_attempts"] == 0
    assert result["accepted_reward"]["mean"] == .25
    assert result["player_tokens"]["completion_tokens"] == 10
    assert result["missing_latency_measurements"] == result["missing_token_measurements"] == 0
    empty = summarize_evaluation([placeholder])
    assert empty["acceptance"]["attempted"] == 0
    assert empty["acceptance"]["rate"] is None
    assert empty["not_started_attempt_slots"] == 1
