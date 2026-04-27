def benchmark_full_active_learning_loop():
    import numpy as np
    from ase.build import molecule
    from ase.calculators.emt import EMT

    from models.descriptors import SOAPDescriptor
    from models.gp_model import AtomicGaussianProcess
    from dft.run_dft import run_dft
    from md.dynamics import run_velocity_verlet_md
    from active_learning.gp_cycle import run_gp_active_learning_cycle

    print("\n=== FULL LOOP BENCHMARK START ===")

    # ---------------------------
    # 1. Build simple system
    # ---------------------------
    atoms = molecule("CH4")

    descriptor = SOAPDescriptor(
        species=["C", "H"],
        cutoff_radius=5.0,
        n_radial=8,
        n_angular=6,
        periodic=False,
    )

    gp_model = AtomicGaussianProcess()

    # ---------------------------
    # 2. Initial dataset (perturbations)
    # ---------------------------
    dataset_structures = [atoms.copy()]

    for shift in [0.03, -0.03, 0.06, -0.06]:
        a = atoms.copy()
        a.positions[1, 0] += shift
        dataset_structures.append(a)

    dataset_energies = []
    for a in dataset_structures:
        E, _ = run_dft(a, EMT())
        dataset_energies.append(E)

    print(f"Initial dataset size: {len(dataset_structures)}")

    # ---------------------------
    # 3. Fit GP
    # ---------------------------
    X_list = [descriptor.create(a) for a in dataset_structures]
    gp_model.fit(X_list, dataset_energies)

    # ---------------------------
    # 4. Run initial MD
    # ---------------------------
    traj, energies = run_velocity_verlet_md(
        atoms,
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=5,
    )

    print("Initial MD energies:", energies)

    # ---------------------------
    # 5. Active learning selection
    # ---------------------------
    results = run_gp_active_learning_cycle(
        structures=traj,
        descriptor=descriptor,
        gp_model=gp_model,
        calculator=EMT(),
        uncertainty_threshold=0.01,
        max_dft_calls=2,
        mode="batch",
        selection_strategy="top_uncertainty",
    )

    selected_structures = results["selected_structures"]
    selected_energies = results["dft_energies"]

    print("Selected indices:", results["selected_indices"])

    # ---------------------------
    # 6. Dataset update
    # ---------------------------
    before = len(dataset_structures)

    dataset_structures.extend(selected_structures)
    dataset_energies.extend(selected_energies)

    after = len(dataset_structures)

    print(f"Dataset size: {before} → {after}")

    # ASSERT: dataset grew
    assert after > before, "Dataset did not grow"

    # ---------------------------
    # 7. Retrain GP
    # ---------------------------
    X_list = [descriptor.create(a) for a in dataset_structures]
    gp_model.fit(X_list, dataset_energies)

    print("GP retrained")

    # ---------------------------
    # 8. Run MD again
    # ---------------------------
    traj2, energies2 = run_velocity_verlet_md(
        atoms,
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=5,
    )

    print("Post-AL MD energies:", energies2)

    # ASSERT: MD still valid
    assert len(traj2) == 6
    assert np.all(np.isfinite(energies2))

    # ---------------------------
    # 9. Check model changed
    # ---------------------------
    if np.allclose(energies, energies2):
        print("WARNING: energies unchanged after AL (possible weak update)")
    else:
        print("SUCCESS: energies changed after AL")

    print("=== FULL LOOP BENCHMARK END ===\n")
