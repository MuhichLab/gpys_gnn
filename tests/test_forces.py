def test_forces_nonzero():
    from md.forces import compute_forces_fd

    F = compute_forces_fd(dataset_structures[0], descriptor, gp_model)

    assert np.all(np.isfinite(F))
    assert F.shape[1] == 3 
