"""Utility functions for computing energies from atomistic models."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms


def compute_energy(atoms: Atoms, descriptor: Any, model: Any) -> float:
    """Compute scalar energy for a structure using a descriptor + model.

    Parameters
    ----------
    atoms
        ASE Atoms object.
    descriptor
        Object with method `create(atoms)` → np.ndarray.
    model
        Model with method `predict_mean([X])`.

    Returns
    -------
    float
        Predicted energy.
    """
    X = descriptor.create(atoms)
    energy = model.predict_mean([X])[0]
    return float(energy)

def compute_energy_batch(atoms_list, descriptor, model):
    X_list = [descriptor.create(a) for a in atoms_list]
    return model.predict_mean(X_list)
