"""Portable evaluation summaries, with explicit sample counts and uncertainty."""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any, Iterable


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def measurements(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if _finite(value)]
    if not numbers:
        return {"n": 0, "mean": None, "standard_error": None, "mean_ci95": None}
    average = mean(numbers)
    error = stdev(numbers) / math.sqrt(len(numbers)) if len(numbers) > 1 else None
    return {"n": len(numbers), "mean": average, "standard_error": error,
            "mean_ci95": [average - 1.96 * error, average + 1.96 * error] if error is not None else None,
            "interval_method": "normal approximation; independent episodes assumed"}


def acceptance_interval(accepted: int, attempted: int) -> dict[str, Any]:
    if not attempted:
        return {"accepted": accepted, "attempted": attempted, "rate": None, "ci95": None}
    proportion, z = accepted / attempted, 1.959963984540054
    denominator = 1 + z * z / attempted
    center = (proportion + z * z / (2 * attempted)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / attempted + z * z / (4 * attempted ** 2)) / denominator
    return {"accepted": accepted, "attempted": attempted, "rate": proportion,
            "ci95": [max(0., center - radius), min(1., center + radius)], "interval_method": "Wilson"}


def summarize_evaluation(outcomes: Iterable[dict[str, Any]], *,
                         discarded_outcomes: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    eligible_slots = list(outcomes)
    discarded_slots = list(discarded_outcomes)
    # Provider-down placeholders retain their scoreboard slots, but no episode
    # ran and therefore no policy acceptance, token cost or latency was observed.
    eligible = [row for row in eligible_slots if row.get("episode_not_started") is not True]
    discarded = [row for row in discarded_slots if row.get("episode_not_started") is not True]
    rows = eligible + discarded
    accepted = [row for row in eligible if row.get("accepted_count") == 1]
    rewards = [((row.get("audit") or {}).get("final_task") or {}).get("reward") for row in accepted]
    usage = [row.get("token_usage") for row in rows]
    token_rows = [row for row in usage if isinstance(row, dict)]
    totals = {key: sum(row.get(key, 0) for row in token_rows if _finite(row.get(key, 0)))
              for key in ("prompt_tokens", "completion_tokens", "thinking_tokens", "response_tokens", "forced_close_tokens")}
    latency = measurements(row.get("wall_seconds") for row in rows)
    return {"acceptance": acceptance_interval(len(accepted), len(rows)),
            "discarded_attempts": len(discarded),
            "not_started_attempt_slots": len(eligible_slots) + len(discarded_slots) - len(rows),
            "accepted_reward": measurements(rewards), "episode_latency_seconds": latency,
            "player_tokens": totals, "episodes_with_token_measurements": len(token_rows),
            "missing_token_measurements": len(rows) - len(token_rows),
            "missing_latency_measurements": len(rows) - latency["n"],
            "reward_population": "accepted episodes only; rejected attempts remain in acceptance rate"}
