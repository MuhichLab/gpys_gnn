def test_active_learning_selects():
    from active_learning.gp_cycle import run_gp_active_learning_cycle
    from ase.calculators.emt import EMT

    traj, _ = run_velocity_verlet_md(
        dataset_structures[0],
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
