"""Descriptor utilities for atomistic structures."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


def infer_species_from_structures(structures):
    """Infer sorted unique chemical species from a list of ASE Atoms."""
    species = set()
    for atoms in structures:
        species.update(atoms.get_chemical_symbols())
    return sorted(species)



class SOAPDescriptor:
    """Simple wrapper around dscribe's SOAP descriptor.

    Parameters
    ----------
    species
        List of chemical species symbols used to define the descriptor space.
    cutoff_radius
        Radial cutoff in angstrom for local environment construction.
    n_radial
        Number of radial basis functions.
    n_angular
        Maximum angular momentum quantum number.
    periodic
        Whether to use periodic boundary conditions when computing SOAP.
    """

    def __init__(
        self,
        species: Sequence[str],
        cutoff_radius: float = 5.0,
        n_radial: int = 8,
        n_angular: int = 6,
        periodic: bool = False,
    ) -> None:
        self.species = list(species)
        self.cutoff_radius = cutoff_radius
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.periodic = periodic

        self._soap = SOAP(
            species=self.species,
            r_cut=self.cutoff_radius,
            n_max=self.n_radial,
            l_max=self.n_angular,
            average="off",
            periodic=self.periodic,
            sparse=False,
        )


    @property
    def n_features(self):
        return self._soap.get_number_of_features()

    def create(self, atoms: Atoms) -> np.ndarray:
        """Create per-atom SOAP features for a single structure.

        Returns
        -------
        np.ndarray
            Array with shape ``(n_atoms, n_features)``.
        """
        if self.periodic and not np.any(atoms.pbc): 
            raise ValueError("Periodic Soap reqeused but Aatoms object has no PBC set.")

        descriptor = self._soap.create(atoms)
        return np.asarray(descriptor)

    def create_batch(self, list_of_atoms):
        descriptors = []
        structure_ids = []

        for i, atoms in enumerate(list_of_atoms):
            d = self.create(atoms)
            descriptors.append(d)
            structure_ids.extend([i] * len(d))

        if not descriptors:
            n_features = self._soap.get_number_of_features()
            return np.empty((0, n_features)), np.array([])

        return np.vstack(descriptors), np.array(structure_ids)

    def _check_species(self, atoms: Atoms):
    	unique = set(atoms.get_chemical_symbols())
    	missing = unique - set(self.species)
    	if missing:
        	raise ValueError(f"Found species not in descriptor: {missing}")
