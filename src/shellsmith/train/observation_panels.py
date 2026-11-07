"""Purpose: Build the operator view with existing Backprop table and plot types.
Input: Durable run observations and trainer state.
Output: Six primary panels plus searchable provenance tables.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

from shellsmith.episode.gm_process.usage import TOKEN_FIELDS

PRIMARY_PANELS = (
    "train/00_run_status", "train/01_operational_timeline", "train/02_update_progress",
    "train/03_learning_metrics", "train/04_episode_progress", "train/05_gm_token_usage",
)


def build_panels(observations: Any, sdk: Any, *, plots: bool = True) -> dict:
    """Purpose: Describe progress without a user-specific dashboard layout.
    Input: The local observation reducers and the selected SDK.
    Output: Automatic table and Plotly panels, with explicit missing-data states.
    """
    with observations.lock:
        state = observations.state
        records = list(observations.records)
        context = records[-1] if records else observations.context()
        at = context["run/observed_at"]
        health = next((row["metrics"] for row in reversed(records) if row["kind"] == "health"), {})
        errors = sum(health.get(key, 0) or 0 for key in (
            "telemetry/collector_errors", "telemetry/logging_errors", "telemetry/sdk_upload_error"))
        condition = "degraded" if errors else "reported healthy"
        if health.get("telemetry/sdk_health_available") == 0:
            condition = "SDK upload health unavailable"
        status = [
            ["Run", observations.config.run.run_id],
            ["Outcome", state.get("status", "running")],
            ["Phase", observations.phase],
            ["Phase duration (seconds)", max(0, context["run/elapsed_seconds"] - observations.phase_started)],
            ["Committed updates / target", f"{state.get('rft_updates_completed', 0)} / {observations.config.train.steps}"],
            ["Latest checkpoint", (state.get("checkpoints") or [{}])[-1].get("path", "none yet")],
            ["Published policy", state.get("active_policy_ref", "none yet")],
            ["Observation interval (seconds)", observations.config.telemetry.system_interval_seconds],
            ["Telemetry", f"{condition} as of {at}"],
            ["Upload backlog", health.get("telemetry/unacknowledged_events", "unavailable")],
            ["Collector errors", health.get("telemetry/collector_errors", 0)],
            ["SDK publication errors", health.get("telemetry/logging_errors", 0)],
            ["Deferred records", health.get("telemetry/deferred_events", 0)],
            ["Inference coverage", "observed counters; gaps are unknown, never zero"],
            ["Last completed training work", health.get("compute/last_completed_phase", "not observed")],
        ]
        counts = observations.episode_counts()
        episodes = [[stage, counts.get(stage, 0), at] for stage in (
            "gm", "player", "judging", "settling", "accepted", "discarded", "rejected", "failed")]
        failures = Counter(row.get("failure_category", "unknown") for row in observations.episodes.values()
                           if row.get("stage") == "failed")
        episodes.extend([["failure: " + key, count, at] for key, count in sorted(failures.items())])
        gm = [[key, observations.gm.get(key), "provider-reported cumulative thread deltas", at]
              for key in TOKEN_FIELDS]
        gm.extend([[key, observations.gm.get(key, 0), "invocations", at] for key in (
            "invocations_started", "invocations_with_usage", "invocations_pending",
            "invocations_unknown", "accounting_errors")])
        gm.extend([["stage: " + key, value["invocations"],
                    f"{value['reported']} with reported usage", at]
                   for key, value in observations.gm.get("stages", {}).items()])
        table = lambda columns, rows: sdk.Table(columns=columns, data=rows, allow_mixed_types=True)
        figures = (_timeline(records), _update_progress(observations), _learning(observations))
        values = {
            PRIMARY_PANELS[0]: table(["Item", "Value"], status + [["Observed at UTC", at]]),
            PRIMARY_PANELS[4]: table(["Episode state", "Count", "Observed at UTC"], episodes),
            PRIMARY_PANELS[5]: table(["Token category / coverage", "Value", "Meaning", "Observed at UTC"], gm),
            "gm/invocations": table(["Invocation", "Thread", "Stage", "Outcome", "Coverage", *TOKEN_FIELDS], observations.gm_rows),
            "episodes/details": table(["Episode", "World", "Outcome", "Reward", "Failure", "Trajectory key"],
                                      [[key, row.get("world_id"), row.get("stage"), row.get("reward"),
                                        row.get("failure_category"), "rollouts/trajectories"]
                                       for key, row in observations.episodes.items()]),
            "updates/details": table(["Update", "Committed count", "Worlds", "Training examples", "Checkpoint"],
                                     [[key, row.get("optimizer_updates_committed"),
                                       json.dumps([world.get("world_id") for world in row.get("worlds", [])]),
                                       row.get("datums"), row.get("sampler_checkpoint")]
                                      for key, row in observations.updates.items()]),
            "updates/publications": table(["Publication", "Committed count", "Kind", "Policy"],
                                          [[key, row.get("step"), row.get("kind", "policy"), row.get("policy_ref")]
                                           for key, row in observations.publications.items()]),
        }
        for key, figure in zip(PRIMARY_PANELS[1:4], figures):
            values[key] = sdk.Plotly(figure) if plots else table(["Figure specification"], [[json.dumps(figure)]])
        return values


def bucket_health(records: list[dict], *, limit: int = 1000) -> list[dict]:
    """Purpose: Bound figure size with time-weighted rates and observed peaks.
    Input: Full-resolution health observations and the bucket limit.
    Output: At most limit points. A bucket with a missing sample retains a gap.
    """
    rows = [row for row in records if row["kind"] == "health"]
    if len(rows) <= limit:
        return rows
    span = max(rows[-1]["run/elapsed_seconds"], 1)
    buckets = {}
    for row in rows:
        index = min(limit - 1, int(row["run/elapsed_seconds"] / span * limit))
        buckets.setdefault(index, []).append(row)
    result = []
    for members in buckets.values():
        row = {**members[-1], "metrics": dict(members[-1]["metrics"])}
        metrics = row["metrics"]
        for key in ("inference/active_requests", "inference/queued_requests"):
            values = [member["metrics"].get(key) for member in members]
            metrics[key] = max(values) if all(value is not None for value in values) else None
        durations = [member["metrics"].get("inference/interval_seconds") for member in members]
        rate = "inference/interval_output_tokens_per_second"
        if all(value is not None and value > 0 for value in durations):
            seconds = sum(durations)
            metrics[rate] = sum(member["metrics"].get("inference/interval_generated_tokens", 0)
                                for member in members) / seconds
            metrics["inference/interval_seconds"] = seconds
        else:
            metrics[rate] = None
        result.append(row)
    return result


def _timeline(records: list[dict]) -> dict:
    rows = bucket_health(records)
    x = [row["run/elapsed_seconds"] for row in rows]
    traces = []
    for label, key in (("Run active requests (bucket peak)", "inference/active_requests"),
                       ("Run queued requests (bucket peak)", "inference/queued_requests"),
                       ("Service capacity", "inference/service_capacity")):
        traces.append({"type": "scatter", "mode": "lines", "name": label,
                       "x": x, "y": [row["metrics"].get(key) for row in rows],
                       "line": {"shape": "hv"}, "connectgaps": False})
    traces.append({"type": "scatter", "mode": "lines", "name": "Aggregate run output tokens/s",
                   "x": x, "y": [row["metrics"].get("inference/interval_output_tokens_per_second") for row in rows],
                   "customdata": [row["metrics"].get("inference/interval_seconds") for row in rows],
                   "hovertemplate": "%{y:.2f} output tokens/s over %{customdata:.2f} seconds<extra></extra>",
                   "yaxis": "y2", "xaxis": "x2", "line": {"shape": "vh"},
                   "connectgaps": False})
    shapes = []
    colors = {"collection": "#6ac9d0", "forward_backward": "#b49ae8", "optimizer": "#fbb360",
              "checkpoint": "#8abbff", "publication": "#4c90f0", "not_observed": "#de7272"}
    end = records[-1]["run/elapsed_seconds"] if records else 0
    for phase, start, finish in _phase_intervals(records):
        color = colors.get(phase, "#89939e")
        label = phase.replace("_", " ")
        # Short phases retain hover detail without text over adjacent labels.
        text = label if finish - start >= max(end, 1) * .12 else ""
        traces.append({"type": "bar", "orientation": "h", "x": [finish - start],
                       "base": [start], "y": ["Trainer phase"], "xaxis": "x3",
                       "yaxis": "y3", "showlegend": False, "marker": {"color": color},
                       "text": [text], "textposition": "inside", "constraintext": "inside",
                       "customdata": [[label, start, finish]],
                       "hovertemplate": "%{customdata[0]}: %{customdata[1]:.1f}–%{customdata[2]:.1f} elapsed seconds<extra></extra>"})
        shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": start, "x1": finish,
                       "y0": 0, "y1": .83, "fillcolor": color,
                       "opacity": .12, "line": {"width": 0}, "layer": "below"})
    return {"data": traces, "layout": {
        "title": {"text": "Inference and trainer phases", "font": {"size": 14}},
        "height": 540, "margin": {"t": 50, "b": 110, "l": 65, "r": 20},
        "legend": {"orientation": "h", "x": 0, "y": -.25, "font": {"size": 10}},
        "xaxis": {"showticklabels": False, "range": [0, max(end, 1)], "anchor": "y"},
        "xaxis2": {"title": "Run elapsed time (seconds)", "matches": "x", "anchor": "y2"},
        "xaxis3": {"matches": "x", "anchor": "y3", "showticklabels": False},
        "yaxis": {"title": "Requests / capacity", "domain": [.48, .83], "rangemode": "tozero", "dtick": 1},
        "yaxis2": {"title": "Output tokens / second", "domain": [0, .35], "rangemode": "tozero"},
        "yaxis3": {"domain": [.9, 1], "visible": False, "fixedrange": True},
        "barmode": "overlay", "shapes": shapes, "hovermode": "x unified"}}


def _phase_intervals(records: list[dict]) -> list[tuple[str, float, float]]:
    """Purpose: Join repeated phase receipts and retain explicit unobserved gaps.
    Input: Ordered durable observations.
    Output: Nonempty elapsed-time intervals with a trainer phase label.
    """
    phases = [row for row in records if row["kind"] in {"phase", "gap"}]
    end = records[-1]["run/elapsed_seconds"] if records else 0
    intervals = []
    for index, row in enumerate(phases):
        start = row.get("gap_started", row["run/elapsed_seconds"])
        finish = (phases[index + 1].get("gap_started", phases[index + 1]["run/elapsed_seconds"])
                  if index + 1 < len(phases) else end)
        if row["kind"] == "gap":
            finish = row["run/elapsed_seconds"]
        if finish <= start:
            continue
        if intervals and intervals[-1][0] == row["phase"] and intervals[-1][2] == start:
            intervals[-1] = (row["phase"], intervals[-1][1], finish)
        else:
            intervals.append((row["phase"], start, finish))
    return intervals


def _update_progress(observations: Any) -> dict:
    rows = bucket_health(observations.records)
    markers = [row for row in observations.records if row["kind"] == "publication"]
    return {"data": [
        {"type": "scatter", "mode": "lines", "name": "Committed optimizer updates",
         "x": [row["run/elapsed_seconds"] for row in rows],
         "y": [row["train/optimizer_updates_committed"] for row in rows], "line": {"shape": "hv"}},
        {"type": "scatter", "mode": "markers", "name": "Checkpoint / publication",
         "x": [row["run/elapsed_seconds"] for row in markers],
         "y": [row["train/optimizer_updates_committed"] for row in markers]}],
        "layout": {"title": f"Optimizer progress — target {observations.config.train.steps}",
                   "xaxis": {"title": "Run elapsed time (seconds)"},
                   "yaxis": {"title": "Committed updates", "dtick": 1, "range": [0, observations.config.train.steps + .2]}}}


def _learning(observations: Any) -> dict:
    """Purpose: Show committed learning results or an explicit pre-update state.
    Input: The current optimizer revision and durable update observations.
    Output: Three aligned metric traces, with a readable constant learning rate.
    """
    # A restored optimizer can invalidate later observed attempts. Keep their
    # evidence in the detail table, but plot only the latest current revision.
    current = observations.state.get("rft_updates_completed", 0)
    revisions = {}
    for row in observations.updates.values():
        count = row.get("optimizer_updates_committed", 0)
        if row.get("datums") and 0 < count <= current:
            revisions[count] = row
    rows = [revisions[count] for count in sorted(revisions)]
    if not rows:
        return {"data": [], "layout": {
            "title": "Learning metrics per committed optimizer update",
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "annotations": [
                {"text": "No optimizer update committed yet", "xref": "paper", "yref": "paper",
                 "x": .5, "y": .6, "showarrow": False, "font": {"size": 16}},
                {"text": f"Current phase: {observations.phase} · 0 / {observations.config.train.steps} committed updates",
                 "xref": "paper", "yref": "paper", "x": .5, "y": .4, "showarrow": False},
            ]}}
    data = []
    for index, (label, section, key) in enumerate((("Loss", "forward_metrics", "loss"),
                                                  ("Gradient norm", "optim_metrics", "gradient_norm"))):
        data.append({"type": "scatter", "mode": "lines+markers", "name": label,
                     "x": [row["optimizer_updates_committed"] for row in rows],
                     "y": [row.get(section, {}).get(key) for row in rows],
                     "yaxis": "y" if index == 0 else "y2", "connectgaps": False})
    data.append({"type": "scatter", "mode": "lines+markers", "name": "Learning rate",
                 "x": [row["optimizer_updates_committed"] for row in rows],
                 "y": [row.get("optim_metrics", {}).get("learning_rate", observations.config.train.learning_rate)
                       for row in rows], "yaxis": "y3"})
    rate_axis = {"title": {"text": "Learning rate", "font": {"size": 10}},
                 "domain": [0, .28], "tickformat": ".1e"}
    reported_rates = [value for value in data[-1]["y"] if value is not None]
    if reported_rates and min(reported_rates) == max(reported_rates):
        rate = reported_rates[0]
        # Plotly's default constant-value range hides small positive rates at zero.
        rate_axis["range"] = [rate * .9, rate * 1.1] if rate > 0 else [0, 1e-8]
    return {"data": data, "layout": {"title": "Learning metrics per committed optimizer update",
            "xaxis": {"title": "Committed optimizer updates", "dtick": 1, "anchor": "y3"},
            "yaxis": {"title": {"text": "Loss", "font": {"size": 10}}, "domain": [.72, 1]},
            "yaxis2": {"title": {"text": "Gradient norm", "font": {"size": 10}}, "domain": [.36, .64]},
            "yaxis3": rate_axis,
            "annotations": []}}
