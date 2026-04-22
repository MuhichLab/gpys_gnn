"""ASE-based DFT execution helpers."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def run_dft(atoms: Any, calculator: Any) -> Tuple[float, np.ndarray]:
    """Run a single-point DFT calculation using ASE calculator interface.

    The input atoms object is not modified. A copy is used internally.
    """
    atoms_copy = atoms.copy()
    atoms_copy.calc = calculator

    energy = float(atoms_copy.get_potential_energy())
    forces = np.asarray(atoms_copy.get_forces(), dtype=float)

    atoms_copy.info["energy"] = energy
    atoms_copy.arrays["forces"] = forces

    return energy, forces

