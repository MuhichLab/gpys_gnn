from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def test_gp_basic(descriptor, gp_model, dataset_structures):
    X_list = [descriptor.create(atoms.copy()) for atoms in dataset_structures]

    mu = gp_model.predict_mean(X_list)
    sigma = gp_model.predict_uncertainty(X_list)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)
