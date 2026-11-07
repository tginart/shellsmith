"""Shared argument resolution for the episode and campaign command lines."""

import argparse

from shellsmith.config import DEFAULT_GM_MODEL, GM_PROVIDER_BY_RUNTIME
from shellsmith.episode.gm_process.shared import validate_gm_runtime


class GMArgumentParser(argparse.ArgumentParser):
    """Reject unsupported GM execution before resolving runtime-dependent defaults."""

    def parse_known_args(self, args=None, namespace=None):
        parsed, remaining = super().parse_known_args(args, namespace)
        if hasattr(parsed, "gm_runtime"):
            try:
                validate_gm_runtime(parsed.gm_runtime)
            except RuntimeError as error:
                self.error(str(error))
        if getattr(parsed, "gm_provider", None) is None and hasattr(parsed, "gm_runtime"):
            parsed.gm_provider = GM_PROVIDER_BY_RUNTIME[parsed.gm_runtime]
        if hasattr(parsed, "gm_model") and parsed.gm_model is None:
            if parsed.gm_runtime != "codex":
                self.error("--gm-model is required for non-Codex runtimes; choose a model supported by that provider")
            parsed.gm_model = DEFAULT_GM_MODEL
        return parsed, remaining
