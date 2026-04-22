"""Gaussian Process-driven active learning loop utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from dft.run_dft import run_dft


def compute_gp_prediction_from_descriptor(x, , gp_model: Any) -> Tuple[float, float]:
    """Compute GP energy mean and uncertainty for one structure."""
    energy = float(np.asarray(gp_model.predict_mean([x])).reshape(-1)[0])
    uncertainty = float(np.asarray(gp_model.predict_uncertainty([x])).reshape(-1)[0])
    return energy, uncertainty

def _pool_descriptor(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=0)


def _flatten_descriptor(x: Any) -> np.ndarray:
    """Flatten descriptor representation into a 1D vector for distance metrics."""
    return np.asarray(x, dtype=float).reshape(-1)


def _apply_dft_on_selected(
    structures: List[Any],
    selected_indices: np.ndarray,
    calculator: Any,
) -> Tuple[List[Any], np.ndarray, List[np.ndarray]]:
    """Run DFT on selected structures and collect outputs."""
    selected_structures: List[Any] = []
    dft_energies: List[float] = []
    dft_forces: List[np.ndarray] = []

    for idx in selected_indices:
        structure_copy = structures[idx].copy()
        energy_dft, forces_dft = run_dft(structure_copy, calculator)

        selected_structures.append(structure_copy)
        dft_energies.append(float(energy_dft))
        dft_forces.append(np.asarray(forces_dft, dtype=float))

    return selected_structures, np.asarray(dft_energies, dtype=float), dft_forces


def _select_farthest_point_indices(
    candidate_indices: np.ndarray,
    descriptor_matrix: np.ndarray,
    uncertainties: np.ndarray,
    max_dft_calls: int,
) -> np.ndarray:
    """Greedy farthest-point sampling seeded by highest uncertainty point."""
    if max_dft_calls <= 0:
        return np.asarray([], dtype=int)

    selected: List[int] = []
    remaining = candidate_indices.copy()

    seed_local = int(np.argmax(uncertainties[remaining]))
    seed_index = int(remaining[seed_local])
    selected.append(seed_index)
    remaining = remaining[remaining != seed_index]

    while len(selected) < max_dft_calls and remaining.size > 0:
        selected_descriptors = descriptor_matrix[np.asarray(selected, dtype=int)]
        remaining_descriptors = descriptor_matrix[remaining]

        pairwise_distances = np.linalg.norm(
            remaining_descriptors[:, None, :] - selected_descriptors[None, :, :],
            axis=2,
        )
        min_distances = pairwise_distances.min(axis=1)
        farthest_local = int(np.argmax(min_distances))
        farthest_index = int(remaining[farthest_local])

        selected.append(farthest_index)
        remaining = remaining[remaining != farthest_index]

    return np.asarray(selected, dtype=int)


def _select_batch_indices(
    gp_uncertainties: np.ndarray,
    descriptor_matrix: np.ndarray,
    uncertainty_threshold: float,
    max_dft_calls: Optional[int],
    selection_strategy: str,
) -> np.ndarray:
    """Select batch-mode DFT indices with optional capped strategy."""
    above_threshold = np.where(gp_uncertainties > uncertainty_threshold)[0]

    if max_dft_calls is None or above_threshold.size <= max_dft_calls:
        return above_threshold

    if selection_strategy == "top_uncertainty":
        order = np.argsort(gp_uncertainties[above_threshold])[::-1]
        return above_threshold[order[:max_dft_calls]]

    if selection_strategy == "random_above_threshold":
        rng = np.random.default_rng()
        selected = rng.choice(above_threshold, size=max_dft_calls, replace=False)
        return np.sort(selected)

    if selection_strategy == "farthest_point":
        return _select_farthest_point_indices(
            above_threshold,
            descriptor_matrix,
            gp_uncertainties,
            max_dft_calls,
        )

    raise ValueError(
        "selection_strategy must be 'top_uncertainty', 'random_above_threshold', "
        "or 'farthest_point'."
    )


def run_gp_active_learning_cycle(
    structures: Iterable[Any],
    descriptor: Any,
    gp_model: Any,
    calculator: Any,
    uncertainty_threshold: float,
    max_dft_calls: Optional[int] = None,
    mode: str = "batch",
    selection_strategy: str = "top_uncertainty",
) -> Dict[str, Any]:
    """Run a GP-based active learning cycle in batch or online mode.

    Parameters
    ----------
    structures
        Iterable of ASE Atoms objects.
    descriptor
        Descriptor object implementing ``create(atoms)``.
    gp_model
        Fitted GP-like model implementing ``predict_mean([X])`` and
        ``predict_uncertainty([X])``.
    calculator
        ASE calculator used by DFT runner.
    uncertainty_threshold
        Structures with uncertainty above this threshold are selected for DFT.
    max_dft_calls
        Optional upper bound on number of DFT evaluations.
    mode
        Active learning mode, either ``"batch"`` or ``"online"``.
    selection_strategy
        Batch-mode capped selection strategy when more points exceed threshold
        than ``max_dft_calls``. Supported: ``"top_uncertainty"`` (default),
        ``"random_above_threshold"``, ``"farthest_point"``.

    Returns
    -------
    dict
        {
            "selected_structures": [...],
            "dft_energies": np.ndarray,
            "dft_forces": list,
            "gp_energies": np.ndarray,
            "gp_uncertainties": np.ndarray,
            "selected_indices": np.ndarray,
        }
    """
    if uncertainty_threshold < 0:
        raise ValueError("uncertainty_threshold must be non-negative.")
    if max_dft_calls is not None and max_dft_calls < 0:
        raise ValueError("max_dft_calls must be >= 0 when provided.")
    if mode not in {"batch", "online"}:
        raise ValueError("mode must be either 'batch' or 'online'.")

    structures_list = list(structures)
    n_structures = len(structures_list)

    gp_energies = np.empty(n_structures, dtype=float)
    gp_uncertainties = np.empty(n_structures, dtype=float)
    descriptor_matrix = None

    if n_structures == 0:
        return {
            "selected_structures": [],
            "dft_energies": np.asarray([], dtype=float),
            "dft_forces": [],
            "gp_energies": gp_energies,
            "gp_uncertainties": gp_uncertainties,
            "selected_indices": np.asarray([], dtype=int),
        }

    selected_indices: List[int] = []
    dft_energies_online: List[float] = []
    dft_forces_online: List[np.ndarray] = []
    selected_structures_online: List[Any] = []

    for idx, atoms in enumerate(structures_list):
        x = descriptor.create(atoms)
        x_vec = _pool_descriptor(x)

        if mode =="batch" : 
		if descriptor_matrix is None:
            		descriptor_matrix = np.empty((n_structures, x_vec.size), dtype=float)
        	elif x_vec.size != descriptor_matrix.shape[1]:
            		raise ValueError(
                		"All flattened descriptors must have the same length for "
                		"batch farthest-point selection."
            		)
		if not np.all(np.isfinite(x_vec)):
     			raise ValueError("Descriptor contains NaN or inf values.")
		x_vec = x_vec / (np.linalg.norm(x_vec) + 1e-12)
	        descriptor_matrix[idx] = x_vec


        energy_gp, sigma_gp = compute_gp_prediction_from_descriptor(x, gp_model)
        gp_energies[idx] = energy_gp
        gp_uncertainties[idx] = sigma_gp

        if mode == "online" and sigma_gp > uncertainty_threshold:
            if max_dft_calls is None or len(selected_indices) < max_dft_calls:
                selected_indices.append(idx)
                structure_copy = atoms.copy()
                energy_dft, forces_dft = run_dft(structure_copy, calculator)
                selected_structures_online.append(structure_copy)
                dft_energies_online.append(float(energy_dft))
                dft_forces_online.append(np.asarray(forces_dft, dtype=float))

    if mode == "batch":
        selected = _select_batch_indices(
            gp_uncertainties,
            descriptor_matrix,
            uncertainty_threshold,
            max_dft_calls,
            selection_strategy,
        )

        selected_structures, dft_energies, dft_forces = _apply_dft_on_selected(
            structures_list,
            selected,
            calculator,
        )
        selected_indices_array = selected.astype(int)
    else:
        selected_indices_array = np.asarray(selected_indices, dtype=int)
        selected_structures = selected_structures_online
        dft_energies = np.asarray(dft_energies_online, dtype=float)
        dft_forces = dft_forces_online

    print("----- Active Learning Summary -----")
    print(f"Total structures: {n_structures}")
    print(f"Selected for DFT: {len(selected_indices_array)}")

    if len(selected_indices_array) > 0:
    	print(f"Max uncertainty: {gp_uncertainties.max():.4e}")
    	print(f"Mean uncertainty: {gp_uncertainties.mean():.4e}")
    	print(f"Selected indices: {selected_indices_array}")


    return {
        "selected_structures": selected_structures,
        "dft_energies": dft_energies,
        "dft_forces": dft_forces,
        "gp_energies": gp_energies,
        "gp_uncertainties": gp_uncertainties,
        "selected_indices": selected_indices_array,
    }

