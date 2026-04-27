from __future__ import annotations

import pytest

EMT = pytest.importorskip("ase.calculators.emt").EMT

from dft.run_dft import run_dft


def test_learning_effect(descriptor, gp_model, dataset_structures, dataset_energies):
    structures = [atoms.copy() for atoms in dataset_structures]
    energies = list(dataset_energies)

    X_list = [descriptor.create(atoms) for atoms in structures]
    gp_model.fit(X_list, energies)

    test_structure = structures[0].copy()
    test_structure.positions[0, 0] += 0.08

    X_test = descriptor.create(test_structure)
    e_true, _ = run_dft(test_structure, EMT())

    e_pred_before = gp_model.predict_mean([X_test])[0]
    err_before = abs(e_pred_before - e_true)

    new_structure = structures[0].copy()
    new_structure.positions[0, 0] += 0.1
    new_energy, _ = run_dft(new_structure, EMT())

    structures.append(new_structure)
    energies.append(new_energy)

    X_list2 = [descriptor.create(atoms) for atoms in structures]
    gp_model.fit(X_list2, energies)

    e_pred_after = gp_model.predict_mean([X_test])[0]
    err_after = abs(e_pred_after - e_true)

    assert err_after <= err_before
