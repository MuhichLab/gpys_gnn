def test_gp_basic():
    X_list = [descriptor.create(a) for a in dataset_structures]

    mu = gp_model.predict_mean(X_list)
    sigma = gp_model.predict_uncertainty(X_list)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)
