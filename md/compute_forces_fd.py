"""Force calculators for MD."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.compute_energy import compute_energy

def compute_forces_fd(
    atoms: Any,
    descriptor: Any,
    gp_model: Any,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Compute forces with central finite differences on GP-predicted energy."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    n_atoms = len(atoms)
    forces = np.zeros((n_atoms, 3), dtype=float)
    base_positions = atoms.get_positions()

    for i in range(n_atoms):
        for j in range(3):
            displaced_plus = atoms.copy()
            displaced_minus = atoms.copy()

            positions_plus = base_positions.copy()
            positions_minus = base_positions.copy()
            positions_plus[i, j] += epsilon
            positions_minus[i, j] -= epsilon

            displaced_plus.set_positions(positions_plus)
            displaced_minus.set_positions(positions_minus)

            e_plus = compute_energy(displaced_plus, descriptor, gp_model)
            e_minus = compute_energy(displaced_minus, descriptor, gp_model)

            forces[i, j] = -(e_plus - e_minus) / (2.0 * epsilon)

    return forces

