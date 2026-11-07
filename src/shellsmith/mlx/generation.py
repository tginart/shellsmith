"""CPU-only generation limits for the native Qwen text/tool contract."""

REASONING_EFFORTS = ("low", "medium", "xhigh")
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_REASONING_TOKENS = {"low": 2048, "medium": 4096, "xhigh": 8192}
# The admitted Qwen tokenizer encodes LF + </think> + two LFs in three IDs.
# Renderer verifies this before a profile's advertised reserve can be used.
FORCED_CLOSE_TOKENS = 3


def reasoning_effort(settings):
    effort = getattr(settings, "reasoning_effort", DEFAULT_REASONING_EFFORT)
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"Native Qwen reasoning_effort must be one of {REASONING_EFFORTS}.")
    return effort


def plain_budget(settings):
    """Legacy plain_tokens is an optional ceiling; otherwise use response_tokens."""
    legacy = getattr(settings, "plain_tokens", 0)
    return min(settings.response_tokens, legacy) if legacy else settings.response_tokens


def reasoning_budget(settings):
    """Resolve the effort-specific limit, retaining explicit legacy overrides."""
    legacy = getattr(settings, "thinking_tokens", 0)
    effort = reasoning_effort(settings)
    return legacy or getattr(settings, f"reasoning_tokens_{effort}", DEFAULT_REASONING_TOKENS[effort])


def completion_budget(settings, *, close_tokens=FORCED_CLOSE_TOKENS):
    """Advertise enough space for either generation mode, including forced IDs.

    max_tokens=0 derives the reserve from the independent phase limits. A
    positive historical value remains an explicit ceiling on *all* recorded
    completion IDs; request-level ceilings have the same strict semantics.
    """
    reserve = max(reasoning_budget(settings) + close_tokens + settings.response_tokens,
                  plain_budget(settings), settings.compaction_tokens)
    legacy = getattr(settings, "max_tokens", 0)
    return min(reserve, legacy) if legacy else reserve
