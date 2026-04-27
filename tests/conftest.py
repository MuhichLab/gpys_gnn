from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when pytest is launched from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def descriptor(dataset_structures):
    pytest.importorskip("numpy")
    pytest.importorskip("dscribe")
    from models.descriptors import SOAPDescriptor
    from models.descriptors import infer_species_from_structures

    species = infer_species_from_structures(dataset_structures)

    return SOAPDescriptor(
        species=species,
        cutoff_radius=5.0,
        n_radial=8,
        n_angular=6,
        periodic=False,
    )


@pytest.fixture
def base_atoms():
    ase_build = pytest.importorskip("ase.build")
    return ase_build.molecule("CH4")


@pytest.fixture
def dataset_structures(base_atoms):
    structures = [base_atoms.copy()]
    for shift in [0.03, -0.03, 0.06, -0.06]:
        atoms = base_atoms.copy()
        atoms.positions[1, 0] += shift
        structures.append(atoms)
    return list(structures)


@pytest.fixture
def dataset_energies(dataset_structures):
    ase_emt = pytest.importorskip("ase.calculators.emt")
    from dft.run_dft import run_dft

    calculator = ase_emt.EMT()
    energies = []
    for atoms in dataset_structures:
        energy, _ = run_dft(atoms.copy(), calculator)
        energies.append(float(energy))
    return list(energies)


@pytest.fixture
def gp_model(descriptor, dataset_structures, dataset_energies):
    from models.gp_model import AtomicGaussianProcess

    model = AtomicGaussianProcess()
    X_list = [descriptor.create(atoms.copy()) for atoms in dataset_structures]
    model.fit(X_list, list(dataset_energies))  # copy defensively
    return model


@pytest.fixture
def calculator():
    ase_emt = pytest.importorskip("ase.calculators.emt")
    return ase_emt.EMT()
