"""ASE-based DFT execution helpers."""

from __future__ import annotations

from typing import Any, Tuple
from ase import Atoms
import numpy as np


def run_dft(atoms: Any, calculator, directory: Optional[str] = =None) -> Tuple[float, np.ndarray]:
    """Run a single-point DFT calculation using ASE calculator interface.

    The input atoms object is not modified. A copy is used internally.
    """
    atoms_copy = atoms.copy()
    if directory is not None:
    	calculator.directory = directory
    atoms_copy.calc = calculator

    try: 
    	energy = float(atoms_copy.get_potential_energy())
    	forces = np.asarray(atoms_copy.get_forces(), dtype=float)
   except Exception as e: 
	rais RuntimeError(f"DFT calcaution failed: {}")

    return energy, forces

