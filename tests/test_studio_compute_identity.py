"""Verify the source inventory that controls numerical checkpoint compatibility."""

from pathlib import Path

import pytest

from studio_compute import model
from studio_compute.protocol import digest


@pytest.mark.parametrize(
    "changed",
    ["activation_budget.py", "engine.py", "model.py", "numerics/ops.py",
     "numerics/metal/gdn_scan_backward.metal"],
)
def test_compute_source_changes_invalidate_numerical_identity(tmp_path, changed):
    """
    Purpose: Bind admission and numerical source edits to the compute identity.
    Input: A source copy with one changed file.
    Output: The numerical digest changes for that file alone.
    """
    originals = model.numerical_sources()
    assert "activation_budget.py" in originals
    for name in originals:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(model.__file__).parent / name).read_bytes())
    assert model.numerical_sources(tmp_path) == originals
    with (tmp_path / changed).open("ab") as file:
        file.write(b"\n// source identity probe\n")
    altered = model.numerical_sources(tmp_path)
    assert {name for name in altered if altered[name] != originals[name]} == {changed}
    assert digest(altered) != digest(originals)
