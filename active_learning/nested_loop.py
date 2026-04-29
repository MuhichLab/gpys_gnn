from __future__ import annotations

import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT
from typing import Optional
from active_learning.gp_cycle import run_gp_active_learning_cycle
from dft.run_dft import run_dft
from md.md_engine import MDEngine
from models.descriptors import SOAPDescriptor, infer_species_from_structures
from models.gp_model import AtomicGaussianProcess


def perturb(atoms, scale=0.1):
    a = atoms.copy()
    a.positions += np.random.normal(scale=scale, size=a.positions.shape)
    return a


def nested_learning_loop(
    dataset_structures,
    calculator = None,
    n_outer=5,
    n_inner_max=5,
    n_select=5,
    md_steps=20,
    convergence_uncertainty=0.01,
    md_step_growth_rate_frac=0.2,
    init_data_set_energies: Optional[list] = None,
    sd_r=5.0,
    sd_n_rad= 8,
    sd_n_ang = 6,
    sd_perd =False,
):
   
    dataset_structures = list(dataset_structures)

    if calculator is None: 
        calculator = EMT()

    md_step_growth_rate=md_step_growth_rate_frac*md_steps

    if init_data_set_energies is None :
        dataset_energies = []
        for a in dataset_structures:
            E, _ = run_dft(a, calculator)
            dataset_energies.append(E)
    else:
        dataset_energies = init_data_set_energies

    species = infer_species_from_structures(dataset_structures)
    descriptor = SOAPDescriptor(
        species=species,
        cutoff_radius= sd_r,
        n_radial=sd_n_rad,
        n_angular=sd_n_ang,
        periodic=sd_perd,
    )

    gp_model = AtomicGaussianProcess()
    X_list = [descriptor.create(a) for a in dataset_structures]
    gp_model.fit(X_list, dataset_energies)

    md_engine = MDEngine()
    results_log = []

    current_seed = dataset_structures[0].copy()

    for outer in range(n_outer):
        print(f"\n=== Outer region {outer} ===")

        for inner in range(n_inner_max):

            traj = md_engine.run(
                current_seed,
                descriptor=descriptor,
                gp_model=gp_model,
                n_steps=md_steps,
            )

            results = run_gp_active_learning_cycle(
                structures=traj,
                descriptor=descriptor,
                gp_model=gp_model,
                calculator=calculator,
                uncertainty_threshold=convergence_uncertainty,
                max_dft_calls=n_select,
                mode="batch",
                selection_strategy="top_uncertainty",
            )

            selected_structures = results["selected_structures"]
            selected_energies = results["dft_energies"]

            max_unc = float(np.max(results["gp_uncertainties"]))
            mean_unc = float(np.mean(results["gp_uncertainties"]))
            n_selected = len(selected_structures)

            log_entry = {
                "outer": outer,
                "inner": inner,
                "dataset_size": len(dataset_structures),
                "max_uncertainty": max_unc,
                "mean_uncertainty": mean_unc,
                "n_selected": n_selected,
            }

            print(log_entry)
            results_log.append(log_entry)

            # convergence condition
            if n_selected == 0 or max_unc < convergence_uncertainty:
                print(" Converged local region")
                break

            # --- runaway exploration guard ---
            if max_unc > 10.0 :
                print(" runaway uncertainty; stopping this local region")
                break# --- runaway exploration guard ---


            # update dataset
            dataset_structures.extend(selected_structures)
            dataset_energies.extend(selected_energies)

            species = infer_species_from_structures(dataset_structures)
            descriptor.species = species

            X_list = [descriptor.create(a) for a in dataset_structures]
            gp_model.fit(X_list, dataset_energies)

        # move to new region
        if max_unc < 10.0:
            # good region → continue from here
            current_seed = perturb(traj[-1], scale=0.05 + 0.05 * outer)
        else:
            print(" resetting seed due to runaway")
            # fallback: pick a known good structure
            current_seed = dataset_structures[0].copy()
            #current_seed = dataset_structures[np.random.randint(len(dataset_structures))].copy()




        md_steps = int(md_steps + md_step_growth_rate)
        print("Number of steps to run next is: ", md_steps)

    return dataset_structures, dataset_energies, results_log, gp_model
