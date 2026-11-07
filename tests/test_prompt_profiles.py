"""Model-specific world prompt tails must remain narrow and deterministic."""

from shellsmith.episode.world_prompt_profile import (
    resolve_world_prompt_profile,
)


def test_sol_retains_unmodified_canonical_prompt() -> None:
    profile = resolve_world_prompt_profile(runtime="codex", model="gpt-6.1-sol")
    assert profile.name == "canonical"
    assert profile.tail_path is None
    assert profile.tail() == ""


def test_candidate_models_receive_distinct_profiles() -> None:
    muse = resolve_world_prompt_profile(runtime="pi", model="meta/muse-spark-1.1")
    hy3 = resolve_world_prompt_profile(runtime="pi", model="tencent/hy3")
    grok = resolve_world_prompt_profile(runtime="pi", model="x-ai/grok-4.5")
    sonnet = resolve_world_prompt_profile(runtime="claude", model="sonnet")

    assert {muse.name, hy3.name, grok.name, sonnet.name} == {
        "muse_difficulty_v2",
        "hy3_artifact_discipline_v1",
        "grok_calibration_v1",
        "sonnet_difficulty_v2",
    }
    assert all(profile.tail() for profile in (muse, hy3, grok, sonnet))


def test_hy3_profile_requires_artifact_first_without_recalibrating_difficulty() -> None:
    profile = resolve_world_prompt_profile(runtime="pi", model="tencent/hy3")
    tail = " ".join(profile.tail().split())

    assert "create the required response artifact before" in tail
    assert "Preserve the canonical world's realism, difficulty" in tail
    assert "Do not simplify the task" in tail


def test_opus_stays_canonical_until_it_has_observed_calibration_failure() -> None:
    profile = resolve_world_prompt_profile(runtime="claude", model="opus")
    assert profile.name == "canonical"


def test_pi_player_receives_qwen_difficulty_calibration() -> None:
    profile = resolve_world_prompt_profile(runtime="pi", model="qwen/qwen3.8-27b")
    assert profile.name == "qwen_difficulty_v1"
    tail = profile.tail()
    normalized_tail = " ".join(tail.split())
    assert "compact proof chain" in tail
    assert "Broad inspection is a legitimate strategy" in tail
    assert "must not be penalized merely for being thorough" in normalized_tail
    assert "only when it adds no new information" in normalized_tail
    assert "private_world_summary" in normalized_tail
    assert "Do not report status/ready/failed" in normalized_tail
    assert "physical vat is never the virtual filesystem" in normalized_tail


def test_sonnet_profile_masks_physical_network_isolation() -> None:
    profile = resolve_world_prompt_profile(runtime="claude", model="sonnet")
    tail = profile.tail()
    normalized_tail = " ".join(tail.split())

    assert "vat's real subprocess may have no DNS" in normalized_tail
    assert "literal `webFetch` executable" in normalized_tail
    assert "Do not answer with DNS failure" in normalized_tail
    assert "declare a simulator failure and make the episode untrainable" in normalized_tail
    assert "support realistic search requests" in normalized_tail
    assert "Do not skip that work by volunteering the exact paper URL" in normalized_tail
