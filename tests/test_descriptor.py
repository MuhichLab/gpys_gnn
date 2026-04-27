from __future__ import annotations

import pytest

ase_build = pytest.importorskip("ase.build")


import pytest


def test_descriptor_missing_species_fails():
    from ase.build import molecule
    from models.descriptors import SOAPDescriptor

    # Descriptor only knows about C and H
    descriptor = SOAPDescriptor(
        species=["C", "H"],
        cutoff_radius=5.0,
        n_radial=8,
        n_angular=6,
        periodic=False,
    )

    # Water contains O → should fail
    mol = molecule("H2O")

    with pytest.raises(ValueError):
        descriptor.create(mol)


def test_descriptor_shapes():
    from ase.build import molecule
    from models.descriptors import SOAPDescriptor
    from models.descriptors import infer_species_from_structures

    mols = [
        molecule("CH4"),
        molecule("CH3OH"),
        molecule("H2O"),
        molecule("NH3"),
    ]

    species = infer_species_from_structures(mols)

    descriptor = SOAPDescriptor(
        species=species,
        cutoff_radius=5.0,
        n_radial=8,
        n_angular=6,
        periodic=False,
    )

    shapes = [descriptor.create(m).shape[1] for m in mols]

    assert len(set(shapes)) == 1


def test_descriptor_species_consistency(descriptor, dataset_structures):
    inferred = set(descriptor.species)

    for atoms in dataset_structures:
        actual = set(atoms.get_chemical_symbols())
        assert actual.issubset(inferred)
