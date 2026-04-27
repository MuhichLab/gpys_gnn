from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
EMT = pytest.importorskip("ase.calculators.emt").EMT

from active_learning.gp_cycle import run_gp_active_learning_cycle
from md.dynamics import run_velocity_verlet_md


def test_full_cycle(descriptor, gp_model, base_atoms, dataset_structures, dataset_energies):
    structures = [atoms.copy() for atoms in dataset_structures]
    energies = list(dataset_energies)

    initial_size = len(structures)

    traj, md_energies_before = run_velocity_verlet_md(
        base_atoms.copy(),
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=4,
    )

    results = run_gp_active_learning_cycle(
        structures=traj,
        descriptor=descriptor,
        gp_model=gp_model,
        calculator=EMT(),
        uncertainty_threshold=0.0,
        max_dft_calls=2,
        mode="batch",
        selection_strategy="top_uncertainty",
        verbose=False,
    )

    structures.extend([atoms.copy() for atoms in results["selected_structures"]])
    energies.extend([float(e) for e in results["dft_energies"]])

    gp_model.fit([descriptor.create(atoms) for atoms in structures], energies)

    _, md_energies_after = run_velocity_verlet_md(
        base_atoms.copy(),
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=4,
    )

    assert len(structures) > initial_size
    assert np.all(np.isfinite(md_energies_before))
    assert np.all(np.isfinite(md_energies_after))
    assert np.all(np.isfinite(energies))
