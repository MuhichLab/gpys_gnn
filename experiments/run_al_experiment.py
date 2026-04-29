from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT

from models.gp_model import AtomicGaussianProcess
from active_learning.nested_loop import nested_learning_loop


def build_initial_structures():
    from ase.build import molecule
    import numpy as np

    atoms = molecule("CH4")
    atoms.center(vacuum=5.0)

    structures = [atoms]

    # add small perturbations
    for shift in [0.02, -0.02, 0.05, -0.05]:
        a = atoms.copy()
        a.positions[1, 0] += shift
        structures.append(a)

    return structures


def plot_results(results_log):
    cycles = list(range(len(results_log)))
    max_u = [r["max_uncertainty"] for r in results_log]
    mean_u = [r["mean_uncertainty"] for r in results_log]
    sizes = [r["dataset_size"] for r in results_log]

    # Plot uncertainty
    plt.figure()
    plt.plot(cycles, max_u, marker="o", label="Max uncertainty")
    plt.plot(cycles, mean_u, marker="s", label="Mean uncertainty")
    plt.xlabel("Cycle")
    plt.ylabel("Uncertainty")
    plt.title("Active Learning Uncertainty")
    plt.legend()
    plt.grid()
    plt.savefig("uncertainty.png")
    plt.close()

    # Plot dataset growth
    plt.figure()
    plt.plot(cycles, sizes, marker="o")
    plt.xlabel("Cycle")
    plt.ylabel("Dataset size")
    plt.title("Dataset Growth")
    plt.grid()
    plt.savefig("dataset_size.png")
    plt.close()


def positions_to_atoms(seed_atoms, traj_positions):
    frames = []
    for pos in traj_positions:
        a = seed_atoms.copy()
        a.set_positions(pos)
        frames.append(a)
    return frames

def run_experiment(
    n_outer=5,
    n_inner_max=5,
    n_select=5,
    md_steps=100,
    convergence_uncertainty=0.01,
    md_step_growth_rate_frac=0.2
):

    dataset_structures=build_initial_structures()

    dataset_structures, dataset_energies, results_log, gp_model = nested_learning_loop(
        dataset_structures,
        calculator = EMT(),
        n_outer=5,
        n_inner_max=5,
        n_select=5,
        md_steps=20,
        convergence_uncertainty=0.01,
        md_step_growth_rate_frac=0.2,
        sd_r=5.0,
        sd_n_rad= 8,
        sd_n_ang = 6,
        sd_perd =False,
    )


    # save + plot
    with open("results.json", "w") as f:
        json.dump(results_log, f, indent=2)

    plot_results(results_log)

if __name__ == "__main__":
    run_experiment()
