from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from md.compute_forces_fd import compute_forces_fd


def test_forces_nonzero(descriptor, gp_model, dataset_structures):
    forces = compute_forces_fd(dataset_structures[0].copy(), descriptor, gp_model)

    assert np.all(np.isfinite(forces))
    assert forces.shape[1] == 3
