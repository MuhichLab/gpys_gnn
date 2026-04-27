from __future__ import annotations

import pytest

EMT = pytest.importorskip("ase.calculators.emt").EMT

from active_learning.gp_cycle import run_gp_active_learning_cycle
from md.dynamics import run_velocity_verlet_md


def test_active_learning_selects(descriptor, gp_model, dataset_structures):
    traj, _ = run_velocity_verlet_md(
        dataset_structures[0].copy(),
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=5,
    )

    results = run_gp_active_learning_cycle(
        structures=traj,
        descriptor=descriptor,
        gp_model=gp_model,
        calculator=EMT(),
        uncertainty_threshold=0.0,
        max_dft_calls=2,
    )

    assert len(results["selected_structures"]) > 0
