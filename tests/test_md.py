from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from md.dynamics import run_velocity_verlet_md


def test_md_runs(descriptor, gp_model, dataset_structures):
    traj, energies = run_velocity_verlet_md(
        dataset_structures[0].copy(),
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=3,
    )

    assert len(traj) == 4
    assert len(energies) == 4
    assert np.all(np.isfinite(energies))
