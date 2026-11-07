"""Load shellsmith runs into a stable, viewer-only representation.

Architecture map
----------------
``runs root``
    -> recursively find ``episode.json``
    -> join optional run config, final judgment, audit, and turn pairs
    -> return JSON-safe summaries/details to the Flask API

This module deliberately imports no orchestrator code.  A half-written or old run
is still useful while debugging, so optional artifacts remain visibly absent rather
than making the whole catalog disappear.  Malformed JSON is reported as an explicit
run error; it is never silently replaced with data from another file.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonResult:
    value: Any | None
    error: str | None = None


def _read_json(path: Path, *, required: bool = False) -> JsonResult:
    """Read one artifact and preserve a useful error for the dashboard.

    Main callers:
        ``load_catalog`` and ``load_run``.

    Missing optional files mean "not produced yet".  Existing malformed files are
    always errors because showing invented defaults would conceal simulator faults.
    """
    try:
        if any(candidate.is_symlink() for candidate in (path, *path.parents)):
            return JsonResult(None, f"refusing symlinked artifact path: {path.name}")
        if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
            return JsonResult(None, f"refusing non-regular artifact: {path.name}")
        return JsonResult(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        message = f"missing required artifact: {path.name}" if required else None
        return JsonResult(None, message)
    except (OSError, json.JSONDecodeError) as error:
        return JsonResult(None, f"could not read {path.name}: {error}")


def _relative_id(root: Path, run_dir: Path) -> str:
    return run_dir.resolve().relative_to(root.resolve()).as_posix()


def resolve_run_dir(root: Path, run_id: str) -> Path | None:
    """Resolve a catalog ID without permitting traversal outside ``root``."""
    if not run_id or Path(run_id).is_absolute():
        return None
    try:
        candidate = root.joinpath(run_id).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate if candidate.is_dir() and candidate.joinpath("episode.json").is_file() else None


def _integrity_summary(final: dict[str, Any], audit: dict[str, Any]) -> tuple[bool | None, list[str]]:
    issues = list(final.get("integrity_issues") or [])
    section = audit.get("simulator_integrity") or {}
    issues.extend(item for item in (section.get("issues") or []) if item not in issues)
    passed = section.get("passed")
    if isinstance(passed, bool):
        return passed, issues
    if final:
        trainable = final.get("trainable")
        return trainable is True and not issues, issues
    return None, issues


def _find_batch(root: Path, run_dir: Path) -> str | None:
    root = root.resolve()
    run_dir = run_dir.resolve()
    current = run_dir.parent
    while current != root and current.is_relative_to(root):
        if current.joinpath("batch_report.json").is_file():
            return current.relative_to(root).as_posix()
        current = current.parent
    return None


def _find_campaign(root: Path, rollout_dir: Path) -> str | None:
    """Walk up until a campaign scoreboard/manifest is found.

    Main callers:
        ``_summary`` while building the Observatory catalog.

    Why this exists:
        Tryout scripts write one campaign directory with many task/rollout
        children. Operators want that whole directory as one filterable chunk.
        Checkout batches use ``batch_report.json`` instead; those stay on
        ``_find_batch``.
    """
    root = root.resolve()
    rollout_dir = rollout_dir.resolve()
    current = rollout_dir.parent
    while current != root and current.is_relative_to(root):
        if current.joinpath("campaign_report.json").is_file() or current.joinpath(
            "campaign_manifest.json"
        ).is_file():
            return current.relative_to(root).as_posix()
        current = current.parent
    return None


def _find_run_directory(root: Path, rollout_dir: Path) -> str:
    """Return the durable trainer/campaign/batch enclosure that owns one rollout.

    Main callers:
        ``_summary`` while building the Observatory catalog.

    A trainer run is identified by its ``trainer_state.json`` rather than
    by a fixed nesting depth. Campaign tryouts use ``campaign_report.json`` /
    ``campaign_manifest.json``. Checkout batches use ``batch_report.json``.
    Standalone smoke rollouts fall back to their immediate parent.
    """
    root = root.resolve()
    rollout_dir = rollout_dir.resolve()
    current = rollout_dir.parent
    while current.is_relative_to(root):
        if current.joinpath("trainer_state.json").is_file():
            relative = current.relative_to(root).as_posix()
            return root.name if relative == "." else relative
        if current == root:
            break
        current = current.parent
    campaign = _find_campaign(root, rollout_dir)
    if campaign is not None:
        return campaign
    batch = _find_batch(root, rollout_dir)
    if batch is not None:
        return batch
    relative_parent = rollout_dir.parent.relative_to(root).as_posix()
    return root.name if relative_parent == "." else relative_parent


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _reasoning_means_from_audit(audit: dict[str, Any]) -> tuple[float | None, float | None]:
    """Pull mean quality/efficiency from acceptance_audit when present."""
    reasoning = audit.get("reasoning")
    if not isinstance(reasoning, dict):
        return None, None
    quality = reasoning.get("mean_quality")
    efficiency = reasoning.get("mean_efficiency")
    return (
        float(quality) if isinstance(quality, (int, float)) else None,
        float(efficiency) if isinstance(efficiency, (int, float)) else None,
    )


def _reasoning_means_from_turns(turns: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Average per-turn reasoning_assessment scores (detail / incomplete runs)."""
    qualities: list[float] = []
    efficiencies: list[float] = []
    for turn in turns:
        assessment = turn.get("reasoning_assessment")
        if not isinstance(assessment, dict):
            continue
        quality = assessment.get("quality")
        efficiency = assessment.get("efficiency")
        if isinstance(quality, (int, float)):
            qualities.append(float(quality))
        if isinstance(efficiency, (int, float)):
            efficiencies.append(float(efficiency))
    return _mean(qualities), _mean(efficiencies)


def _classify_episode_end(
    *,
    final_request: dict[str, Any],
    last_turn: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify why the episode stopped after the last world turn.

    Main callers:
        ``_summary`` / ``load_run`` for Observatory chips.

    Why this exists:
        GM often leaves ``terminate_episode=false`` on the last turn while
        Player emits a text-only (or malformed-tool) assistant message and Pi
        settles. That is the only case labeled ``agent_terminal`` / dangling.
        World/budget stops may still record leftover assistant text; that text is
        kept on ``agent_final_response`` but is not a dangling stop.
    """
    if not final_request:
        return {
            "end_kind": None,
            "end_label": None,
            "agent_final_response": None,
            "has_dangling_agent_final": False,
            "forced_budget_termination": None,
            "world_termination_reason": None,
            "player_agent_prompt_terminated": None,
        }

    agent_final = final_request.get("player_final_response")
    if not isinstance(agent_final, str):
        agent_final = None
    else:
        agent_final = agent_final.strip() or None

    forced_budget = bool(final_request.get("forced_budget_termination"))
    world_reason = final_request.get("world_termination_reason")
    if world_reason is not None:
        world_reason = str(world_reason)
    prompt_terminated = bool(final_request.get("player_agent_prompt_terminated"))
    last_world_terminate = bool((last_turn or {}).get("terminate_episode"))

    if forced_budget:
        end_kind = "budget_terminal"
        end_label = "budget terminal"
    elif world_reason or last_world_terminate:
        end_kind = "world_terminal"
        end_label = "world terminal"
    elif prompt_terminated:
        end_kind = "vat_terminal"
        end_label = "vat terminal"
    else:
        # Pi settled without world/budget stop → Player's last message had no tools.
        end_kind = "agent_terminal"
        end_label = "agent terminal"

    return {
        "end_kind": end_kind,
        "end_label": end_label,
        "agent_final_response": agent_final,
        # "Dangling" means the episode stopped because Player emitted a no-tool
        # final message. World/budget stops may still leave leftover assistant
        # text after terminate=true; that is not a dangling stop.
        "has_dangling_agent_final": end_kind == "agent_terminal",
        "forced_budget_termination": forced_budget,
        "world_termination_reason": world_reason,
        "player_agent_prompt_terminated": prompt_terminated,
    }


def _policy_provenance(run_dir: Path) -> dict[str, Any]:
    """Read ``policy_provenance.json`` — which weights sampled Player.

    Main callers:
        ``_summary``.

    Why this exists:
        Every trainer episode is stamped with the base/LoRA checkpoint that
        produced its samples. Without it on the dashboard there is no way to tell
        which RFT update a rollout belongs to, so a stale-checkpoint bug looks
        exactly like a good run.
    """
    result = _read_json(run_dir.joinpath("policy_provenance.json"))
    value = result.value if isinstance(result.value, dict) else {}
    return {
        "policy_kind": value.get("policy_kind"),
        "policy_ref": value.get("policy_ref"),
        "policy_rft_updates_completed": value.get("rft_updates_completed"),
    }


def _durable_outcome_view(run_dir: Path) -> dict[str, Any]:
    """Read ``logs/durable_outcome.json`` — the resume-facing job verdict.

    Main callers:
        ``_summary``.

    Why this exists:
        The campaign records three things here that no other artifact carries:
        which GM actually ran the episode (after a provider fallback), and
        the two states in which a rollout directory is *not* a normal result —
        ``episode_not_started`` (the provider was down, nothing ran) and
        ``discarded_by_provider_fallback`` (the episode ran, but its group was
        thrown away and redone on the backup simulator). Both look like ordinary
        low-reward rollouts on the dashboard unless they are surfaced.
    """
    result = _read_json(run_dir.joinpath("logs", "durable_outcome.json"))
    payload = result.value if isinstance(result.value, dict) else {}
    outcome = payload.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else {}
    return {
        "gm_fallback_applied": outcome.get("gm_fallback_applied"),
        "original_gm_model": outcome.get("original_gm_model"),
        "episode_not_started": outcome.get("episode_not_started") is True,
        "discarded_by_provider_fallback": (
            outcome.get("discarded_by_provider_fallback") is True
        ),
        "discard_reason": outcome.get("discard_reason"),
    }


def _run_warnings(
    *, provenance: dict[str, Any], outcome: dict[str, Any], gm_model: str
) -> list[str]:
    """Short human chips for states that would otherwise read as a bad rollout.

    Main callers:
        ``_summary``.
    """
    warnings: list[str] = []
    if outcome["episode_not_started"]:
        warnings.append("never started — GM provider was down")
    if outcome["discarded_by_provider_fallback"]:
        reason = outcome.get("discard_reason") or "provider fallback"
        warnings.append(f"discarded and redone on the backup GM ({reason})")
    if outcome["gm_fallback_applied"] is True:
        original = outcome.get("original_gm_model") or "the primary GM"
        warnings.append(f"ran on backup GM {gm_model} (fell back from {original})")
    if provenance["policy_kind"] == "base" and (
        isinstance(provenance["policy_rft_updates_completed"], int)
        and provenance["policy_rft_updates_completed"] > 0
    ):
        warnings.append(
            "sampled from the base policy after "
            f"{provenance['policy_rft_updates_completed']} RFT update(s)"
        )
    return warnings


def _last_turn_terminate(run_dir: Path) -> dict[str, Any] | None:
    """Load only the last turn response for end-classification (catalog path)."""
    turns_dir = run_dir.joinpath("world", "turns")
    if any(candidate.is_symlink() for candidate in (turns_dir, *turns_dir.parents)):
        return None
    if not turns_dir.is_dir():
        return None
    turn_dirs = sorted(
        (
            path
            for path in turns_dir.iterdir()
            if path.is_dir() and not path.is_symlink() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    if not turn_dirs:
        return None
    response = _read_json(turn_dirs[-1].joinpath("response.json"))
    return response.value if isinstance(response.value, dict) else None


def _summary(root: Path, episode_path: Path) -> dict[str, Any]:
    run_dir = episode_path.parent
    episode_result = _read_json(episode_path, required=True)
    config_result = _read_json(run_dir.joinpath("run_config.json"))
    final_result = _read_json(run_dir.joinpath("world", "final", "response.json"))
    final_request_result = _read_json(run_dir.joinpath("world", "final", "request.json"))
    audit_result = _read_json(run_dir.joinpath("acceptance_audit.json"))
    errors = [
        result.error
        for result in (episode_result, config_result, final_result, audit_result)
        if result.error
    ]
    episode = episode_result.value if isinstance(episode_result.value, dict) else {}
    config = config_result.value if isinstance(config_result.value, dict) else {}
    final = final_result.value if isinstance(final_result.value, dict) else {}
    final_request = (
        final_request_result.value
        if isinstance(final_request_result.value, dict)
        else {}
    )
    audit = audit_result.value if isinstance(audit_result.value, dict) else {}
    integrity_passed, integrity_issues = _integrity_summary(final, audit)
    trainable = final.get("trainable")
    if trainable is None:
        trainable = audit.get("accepted_for_training")
    completed_actions = 0
    turns_dir: Path = run_dir.joinpath("world", "turns")
    turns_path_safe = not any(
        candidate.is_symlink() for candidate in (turns_dir, *turns_dir.parents)
    )
    if turns_path_safe and turns_dir.is_dir():
        completed_actions = sum(
            turn_dir.is_dir()
            and not turn_dir.is_symlink()
            and turn_dir.joinpath("request.json").is_file()
            and turn_dir.joinpath("response.json").is_file()
            for turn_dir in turns_dir.iterdir()
        )
    mean_quality, mean_efficiency = _reasoning_means_from_audit(audit)
    provenance = _policy_provenance(run_dir)
    durable_outcome = _durable_outcome_view(run_dir)
    gm_model = config.get("gm_model") or "unknown"
    end_info = _classify_episode_end(
        final_request=final_request,
        last_turn=_last_turn_terminate(run_dir) if final_request else None,
    )
    agent_final = end_info["agent_final_response"]
    return {
        "id": _relative_id(root, run_dir),
        "episode_id": episode.get("episode_id") or config.get("episode_id") or run_dir.name,
        "task": episode.get("world_id") or config.get("world_file") or "unknown",
        "player_model": config.get("player_model") or "unknown",
        "gm_model": gm_model,
        "run_dir": _find_run_directory(root, run_dir),
        "campaign": _find_campaign(root, run_dir),
        "batch": _find_batch(root, run_dir),
        "status": episode.get("status") or "unknown",
        "action_count": completed_actions,
        "reward": final.get("final_reward"),
        "mean_quality": mean_quality,
        "mean_efficiency": mean_efficiency,
        "completion_percentage": final.get("completion_percentage"),
        "trainable": trainable,
        "integrity_passed": integrity_passed,
        "integrity_issue_count": len(integrity_issues),
        "accepted_for_training": audit.get("accepted_for_training"),
        "end_kind": end_info["end_kind"],
        "end_label": end_info["end_label"],
        "has_dangling_agent_final": bool(end_info["has_dangling_agent_final"]),
        "agent_final_preview": (
            (agent_final[:160] + "…")
            if end_info["has_dangling_agent_final"]
            and agent_final
            and len(agent_final) > 160
            else (agent_final if end_info["has_dangling_agent_final"] else None)
        ),
        **provenance,
        **durable_outcome,
        "warnings": _run_warnings(
            provenance=provenance,
            outcome=durable_outcome,
            gm_model=gm_model,
        ),
        "errors": errors,
    }


def load_catalog(root: Path) -> dict[str, Any]:
    """Build the dashboard catalog from every durable episode below ``root``.

    Main callers:
        ``GET /api/runs``.

    Recursive discovery supports both standalone smoke directories and the nested
    rollout directories produced by parallel batches without coupling to their
    naming conventions.
    """
    if not root.is_dir():
        return {"root": str(root), "runs": [], "error": f"runs root does not exist: {root}"}
    # Resolve the root itself: ``_read_json`` refuses any symlinked path
    # component, and the default root lives under the system temp dir, which on
    # macOS is reached through the ``/var -> /private/var`` symlink. Symlinks
    # *inside* the root are still skipped below and refused by ``_read_json``.
    root = root.resolve()
    episode_paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if not directory_path.joinpath(name).is_symlink()
        ]
        if "episode.json" in file_names:
            episode_path: Path = directory_path.joinpath("episode.json")
            if not episode_path.is_symlink():
                episode_paths.append(episode_path)
    runs = [_summary(root, path) for path in episode_paths]
    runs.sort(key=lambda item: item["id"], reverse=True)
    return {"root": str(root.resolve()), "runs": runs, "error": None}


def _extract_reasoning(request_data: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for event in request_data.get("trajectory_delta") or []:
        message = (event.get("payload") or {}).get("message") or {}
        if message.get("role") != "assistant":
            continue
        for content in message.get("content") or []:
            kind = content.get("type")
            text = content.get("thinking") if kind == "thinking" else content.get("text")
            if kind in {"thinking", "text"} and text:
                blocks.append({"type": kind, "text": str(text)})
    return blocks


def _extract_user_texts(request_data: dict[str, Any]) -> list[str]:
    """Collect user-role text blocks from a turn's ``trajectory_delta``.

    Main callers:
        ``_load_turns`` / ``_initial_user_message`` for Observatory kickoff display.

    Why this exists:
        The first Pi user message (task brief) is stored inside turn 0's
        trajectory_delta, but Observatory previously only extracted assistant
        reasoning, so the kickoff prompt never appeared in the UI.
    """
    blocks: list[str] = []
    for event in request_data.get("trajectory_delta") or []:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append(text.strip())
    return blocks


def _initial_user_message(turns: list[dict[str, Any]], run_dir: Path) -> str | None:
    """First user message that kicked off Player, if present.

    Prefer turn 0 trajectory_delta (exact Pi kickoff). Fall back to the durable
    ``scenario/player_agent_prompt.md`` when turn 0 has no user text yet.
    """
    for turn in turns:
        user_texts = turn.get("user_texts")
        if isinstance(user_texts, list) and user_texts:
            return "\n\n".join(str(item) for item in user_texts if item)
    prompt_path = run_dir.joinpath("scenario", "player_agent_prompt.md")
    if prompt_path.is_file() and not prompt_path.is_symlink():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def _kickoff_agents_md(run_dir: Path) -> str | None:
    """Vat kickoff ``AGENTS.md`` that Player saw at episode start.

    Main callers:
        ``load_run`` for Observatory trajectory context.

    Why this exists:
        GM's boot writes the vat's only initial file as AGENTS.md. The
        durable audit copy is ``scenario/vat_AGENTS.md``. Observatory needs that
        text beside the first user message so the kickoff context is visible.
    """
    path = run_dir.joinpath("scenario", "vat_AGENTS.md")
    if not path.is_file() or path.is_symlink():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _normalize_actions(request_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer the sibling ``actions`` batch; fall back to singular ``action``.

    Why this exists:
        GM turns often pack several Player tool calls in one resume. The
        durable request keeps both ``action`` (first call) and ``actions``
        (full batch). Observatory must render the batch or it looks empty.
    """
    actions = request_data.get("actions")
    if isinstance(actions, list) and actions:
        return [item for item in actions if isinstance(item, dict)]
    action = request_data.get("action")
    if isinstance(action, dict):
        return [action]
    return []


def _normalize_tool_results(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer ``tool_results``; fall back to singular ``tool_response``.

    Main callers:
        ``_load_turns`` when building Observatory trajectory steps.

    Multi-call turns write ``tool_results: [{action_id, tool_response}, ...]``
    and omit singular ``tool_response``. Reading only the singular field made
    the UI show ``null`` even when every sibling observation was present.
    """
    results = response_data.get("tool_results")
    if isinstance(results, list) and results:
        return [item for item in results if isinstance(item, dict)]
    tool_response = response_data.get("tool_response")
    if tool_response is None and response_data.get("action_id") is None:
        return []
    return [
        {
            "action_id": response_data.get("action_id"),
            "tool_response": tool_response,
        }
    ]


def _load_turns(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    turns: list[dict[str, Any]] = []
    errors: list[str] = []
    turns_dir: Path = run_dir.joinpath("world", "turns")
    if any(candidate.is_symlink() for candidate in (turns_dir, *turns_dir.parents)):
        return turns, ["refusing symlinked world/turns path"]
    if not turns_dir.is_dir():
        return turns, errors
    for turn_dir in sorted(turns_dir.iterdir()):
        if turn_dir.is_symlink():
            errors.append(f"{turn_dir.name}: refusing symlinked turn directory")
            continue
        if not turn_dir.is_dir():
            continue
        request_result = _read_json(turn_dir.joinpath("request.json"), required=True)
        response_result = _read_json(turn_dir.joinpath("response.json"), required=True)
        if request_result.error:
            errors.append(f"{turn_dir.name}: {request_result.error}")
        if response_result.error:
            errors.append(f"{turn_dir.name}: {response_result.error}")
        request_data = request_result.value if isinstance(request_result.value, dict) else {}
        response_data = response_result.value if isinstance(response_result.value, dict) else {}
        actions = _normalize_actions(request_data)
        tool_results = _normalize_tool_results(response_data)
        first_result = tool_results[0] if tool_results else {}
        turns.append(
            {
                "index": request_data.get("step_index", int(turn_dir.name) if turn_dir.name.isdigit() else turn_dir.name),
                "actions": actions,
                "action": actions[0] if actions else None,
                "reasoning": _extract_reasoning(request_data),
                "user_texts": _extract_user_texts(request_data),
                "tool_results": tool_results,
                "tool_response": first_result.get("tool_response"),
                "turn_reward": response_data.get("turn_reward"),
                "difficulty_rating": response_data.get("difficulty_rating"),
                "completion_percentage": response_data.get("completion_percentage"),
                "commentary": response_data.get("commentary"),
                "reasoning_assessment": response_data.get("reasoning_assessment"),
                "trainable": response_data.get("trainable"),
                "integrity_issues": response_data.get("integrity_issues") or [],
                "terminate_episode": response_data.get("terminate_episode"),
                "termination_reason": response_data.get("termination_reason"),
            }
        )
    return turns, errors


def load_run(root: Path, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load one complete viewer record by its safe root-relative catalog ID."""
    run_dir = resolve_run_dir(root, run_id)
    if run_dir is None:
        return None, f"no run {run_id!r} under {root}"
    summary = _summary(root, run_dir.joinpath("episode.json"))
    episode = _read_json(run_dir.joinpath("episode.json"), required=True)
    config = _read_json(run_dir.joinpath("run_config.json"))
    final = _read_json(run_dir.joinpath("world", "final", "response.json"))
    final_request = _read_json(run_dir.joinpath("world", "final", "request.json"))
    audit = _read_json(run_dir.joinpath("acceptance_audit.json"))
    diagnostics = _read_json(run_dir.joinpath("logs", "generation_diagnostics.json"))
    turns, turn_errors = _load_turns(run_dir)
    # Incomplete / mid-run episodes often lack acceptance_audit means; fill from turns.
    if summary.get("mean_quality") is None or summary.get("mean_efficiency") is None:
        turn_quality, turn_efficiency = _reasoning_means_from_turns(turns)
        if summary.get("mean_quality") is None:
            summary["mean_quality"] = turn_quality
        if summary.get("mean_efficiency") is None:
            summary["mean_efficiency"] = turn_efficiency
    final_request_value = (
        final_request.value if isinstance(final_request.value, dict) else {}
    )
    end_info = _classify_episode_end(
        final_request=final_request_value,
        last_turn=turns[-1] if turns else None,
    )
    # Prefer the freshly classified full text over the catalog preview.
    summary["end_kind"] = end_info["end_kind"]
    summary["end_label"] = end_info["end_label"]
    summary["agent_final_response"] = end_info["agent_final_response"]
    summary["has_dangling_agent_final"] = bool(end_info["has_dangling_agent_final"])
    agent_final = end_info["agent_final_response"]
    summary["agent_final_preview"] = (
        (
            agent_final[:160] + "…"
            if end_info["has_dangling_agent_final"]
            and agent_final
            and len(agent_final) > 160
            else (agent_final if end_info["has_dangling_agent_final"] else None)
        )
    )
    if turns and end_info["end_kind"] == "agent_terminal":
        turns[-1] = {**turns[-1], "agent_terminal_stop": True}
    record = {
        "summary": summary,
        "episode": episode.value,
        "run_config": config.value,
        "turns": turns,
        "initial_user_message": _initial_user_message(turns, run_dir),
        "kickoff_agents_md": _kickoff_agents_md(run_dir),
        "terminal_judgment": final.value,
        "final_request": final_request_value or None,
        "acceptance_audit": audit.value,
        "generation_diagnostics": diagnostics.value,
        "policy_provenance": _read_json(
            run_dir.joinpath("policy_provenance.json")
        ).value,
        "durable_outcome": _read_json(
            run_dir.joinpath("logs", "durable_outcome.json")
        ).value,
        "artifact_errors": summary["errors"] + turn_errors + ([diagnostics.error] if diagnostics.error else []),
    }
    return record, None
