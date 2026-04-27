def test_learning_effect():
    import numpy as np
    from ase.calculators.emt import EMT

    # --- copy dataset ---
    structures = list(dataset_structures)
    energies = list(dataset_energies)

    # --- fit initial GP ---
    X_list = [descriptor.create(a) for a in structures]
    gp_model.fit(X_list, energies)

    # --- pick a test structure slightly outside training set ---
    test_structure = structures[0].copy()
    test_structure.positions[0, 0] += 0.08

    X_test = descriptor.create(test_structure)

    # true energy
    E_true, _ = run_dft(test_structure, EMT())

    # prediction before
    E_pred_before = gp_model.predict_mean([X_test])[0]
    err_before = abs(E_pred_before - E_true)

    # --- add new nearby structure ---
    new_structure = structures[0].copy()
    new_structure.positions[0, 0] += 0.1
    new_energy, _ = run_dft(new_structure, EMT())

    structures.append(new_structure)
    energies.append(new_energy)

    # --- retrain ---
    X_list2 = [descriptor.create(a) for a in structures]
    gp_model.fit(X_list2, energies)

    # prediction after
    E_pred_after = gp_model.predict_mean([X_test])[0]
    err_after = abs(E_pred_after - E_true)

    print("Error before:", err_before)
    print("Error after :", err_after)

    # --- ASSERT: learning improves prediction ---
    assert err_after <= err_before
