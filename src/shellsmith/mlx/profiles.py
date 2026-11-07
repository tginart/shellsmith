"""Portable, explicit context candidates; qualification is opt-in by profile.

The production profile remains 32K. Candidate profiles do not claim a passed
qualification: the campaign evidence must establish that separately.
"""

PRODUCTION_PROFILE = "qwen38_dense_8bit_32k"
CONTEXT_STAGES = (32768, 131072, 262144)
QUALIFICATION_PROFILES = {f"qwen38_dense_8bit_{tokens // 1024}k_qualification": tokens
                          for tokens in CONTEXT_STAGES}
PROFILE_CONTEXTS = {PRODUCTION_PROFILE: 32768, **QUALIFICATION_PROFILES}
MEMORY_GIB = 128


def checkpoint_layers(profile):
    if profile not in PROFILE_CONTEXTS:
        raise ValueError(f"Unsupported MLX profile: {profile!r}")
    return 8 if profile in QUALIFICATION_PROFILES else 1


def qualification_profile(tokens):
    if type(tokens) is not int or tokens not in CONTEXT_STAGES:
        raise ValueError("Context qualification stages are exactly 32768, 131072, and 262144 tokens.")
    return f"qwen38_dense_8bit_{tokens // 1024}k_qualification"
