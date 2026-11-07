"""Invalid scalar settings fail before SDK work, snapshots, or rollout collection."""

import asyncio
from dataclasses import replace

import pytest

from shellsmith.campaign.inference import run_inference_campaign
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection, load_train_config, validate_train_config
from shellsmith.train.trainer import run_train


class NoSDK:
    def __getattr__(self, name):
        pytest.fail(f"Invalid configuration reached SDK {name}")


def config(tmp_path, backend):
    return TrainConfig(run=RunSection(run_id="scalar-preflight", log_dir=str(tmp_path / "run")),
                       compute=ComputeSection(backend=backend))


@pytest.mark.parametrize("backend", ["tinker", "studio"])
@pytest.mark.parametrize("values,field", [
    ({"residual_cap_floor": 0.}, "residual_cap_floor"),
    ({"residual_cap": 0., "residual_cap_floor": 0.}, "residual_cap"),
    ({"residual_cap": 0.}, "residual_cap"),
])
def test_zero_residual_caps_fail_direct_and_toml_preflight(tmp_path, backend, values, field):
    base = config(tmp_path, backend)
    invalid = replace(base, credit=replace(base.credit, **values))
    message = field + " must be positive"
    with pytest.raises(ValueError, match=message):
        validate_train_config(invalid)
    with pytest.raises(ValueError, match=message):
        asyncio.run(run_train(invalid, deps=NoSDK()))
    source = tmp_path / "invalid.toml"
    source.write_text(f'[compute]\nbackend = "{backend}"\n[credit]\n' +
                      "".join(f"{key} = {value}\n" for key, value in values.items()))
    with pytest.raises(ValueError, match=message):
        load_train_config(source, log_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob("*.trainer.lock"))


@pytest.mark.parametrize("backend", ["tinker", "studio"])
def test_positive_residual_caps_and_zero_correction_coefficients_are_valid(tmp_path, backend):
    base = config(tmp_path, backend)
    validate_train_config(replace(base, credit=replace(base.credit, residual_cap=.25, residual_cap_floor=.25,
                                                     credit_coefficient=0., reasoning_coefficient=0.)))


@pytest.mark.parametrize("seed", [-1, 2**32])
@pytest.mark.parametrize("inference", [False, True])
def test_studio_seed_rejected_before_run_artifacts_or_discovery(tmp_path, seed, inference):
    base = config(tmp_path, "studio")
    invalid = replace(base, worlds=replace(base.worlds, seed=seed))
    mode = "inference" if inference else "studio_train"
    with pytest.raises(ValueError, match="Studio.*seed.*uint32"):
        validate_train_config(invalid, campaign_mode=mode)
    operation = (run_inference_campaign(invalid, deps=NoSDK(), sampler_path=None)
                 if inference else run_train(invalid, deps=NoSDK()))
    with pytest.raises(ValueError, match="Studio.*seed.*uint32"):
        asyncio.run(operation)
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob("*.trainer.lock"))


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
@pytest.mark.parametrize("mode", ["studio_train", "inference"])
def test_studio_uint32_seed_edges_are_valid(tmp_path, seed, mode):
    base = config(tmp_path, "studio")
    validate_train_config(replace(base, worlds=replace(base.worlds, seed=seed)), campaign_mode=mode)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_studio_seed_constraint_does_not_restrict_tinker_world_selection(tmp_path, seed):
    base = config(tmp_path, "tinker")
    validate_train_config(replace(base, worlds=replace(base.worlds, seed=seed)))
