"""Experimental context admission cannot widen the existing production profile."""

from dataclasses import replace

import pytest

from shellsmith.mlx.profiles import CONTEXT_STAGES, checkpoint_layers, qualification_profile
from shellsmith.train.config import TrainConfig, validate_mlx_config


@pytest.mark.parametrize("tokens", CONTEXT_STAGES)
def test_context_candidate_has_exact_frozen_window_and_recompute_schedule(tokens):
    config = TrainConfig()
    profile = qualification_profile(tokens)
    config = replace(config, mlx=replace(config.mlx, profile=profile, context_window=tokens))
    validate_mlx_config(config)
    assert checkpoint_layers(profile) == 8
    with pytest.raises(ValueError, match="context_window"):
        validate_mlx_config(replace(config, mlx=replace(config.mlx, context_window=tokens + 1)))


@pytest.mark.parametrize("tokens", [65536, 98304, 196608, True])
def test_no_intermediate_qualification_stages(tokens):
    with pytest.raises(ValueError, match="exactly"):
        qualification_profile(tokens)


def test_production_profile_is_still_32k():
    config = TrainConfig()
    assert checkpoint_layers(config.mlx.profile) == 1
    with pytest.raises(ValueError, match="32768"):
        validate_mlx_config(replace(config, mlx=replace(config.mlx, context_window=262144)))
