def test_md_runs():
    from md.dynamics import run_velocity_verlet_md

    traj, energies = run_velocity_verlet_md(
        dataset_structures[0],
        descriptor,
        gp_model,
        timestep=0.05,
        n_steps=3,
    )

    assert len(traj) == 4
    assert len(energies) == 4
    assert np.all(np.isfinite(energies)) 
