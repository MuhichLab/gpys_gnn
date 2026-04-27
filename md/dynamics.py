"""Velocity-Verlet molecular dynamics loop."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from ase import units


from md.compute_forces_fd import compute_forces_fd
from utils.compute_energy import compute_energy




def initialize_velocities(
    atoms: Any,
    temperature: float = 298.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Randomly initialize velocities from a Maxwell-Boltzmann distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    rng = np.random.default_rng(random_state)
    masses = atoms.get_masses()[:, None]

    sigma = np.sqrt(units.kB * temperature / masses)
    velocities = rng.normal(loc=0.0, scale=sigma, size=(len(atoms), 3))
    return velocities


def velocity_scaling_thermostat(
    velocities: np.ndarray,
    masses: np.ndarray,
    target_temperature: float,
) -> np.ndarray:
    """Apply simple velocity-scaling thermostat to target temperature."""
    if target_temperature <= 0:
        raise ValueError("target_temperature must be positive.")

    masses = masses.reshape(-1, 1)
    kinetic_energy = 0.5 * np.sum(masses * velocities**2)
    dof = velocities.size
    if dof == 0:
        raise ValueError("velocities array is empty.")

    current_temperature = (2.0 * kinetic_energy) / (dof * units.kB)
    if current_temperature <= 0:
        raise ValueError("current temperature is non-positive; cannot scale velocities.")

    scale = np.sqrt(target_temperature / current_temperature)
    return velocities * scale


def run_velocity_verlet_md(
    atoms: Any,
    descriptor: Any,
    gp_model: Any,
    timestep: float,
    n_steps: int,
    velocities: Optional[np.ndarray] = None,
    temperature: float = 298.0,
    run_loop: bool = True,
    use_velocity_scaling_thermostat: bool = False,
    fd_epsilon: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run optional MD loop using velocity Verlet.

    Parameters
    ----------
    atoms
        ASE Atoms object (positions and optional velocities).
    descriptor
        Descriptor object used by the GP energy predictor.
    gp_model
        GP model used only for energy predictions.
    timestep
        Time step in femtoseconds.
    n_steps
        Number of integration steps.
    velocities
        Optional initial velocities. If not provided, random velocities are
        initialized from `temperature`.
    temperature
        Temperature in Kelvin used for default velocity initialization and as
        the target for optional velocity-scaling thermostat.
    run_loop
        If False, only evaluate and return the initial state.
    use_velocity_scaling_thermostat
        If True, apply a simple velocity-scaling thermostat each MD step.
    fd_epsilon
        Displacement used by finite-difference force evaluation.

    Returns
    -------
    trajectory, energies
        trajectory: shape (n_recorded_steps, n_atoms, 3)
        energies: shape (n_recorded_steps,)
    """
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0.")
    if timestep <= 0:
        raise ValueError("timestep must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    atoms_md = atoms.copy()

    if velocities is None:
        existing_velocities = atoms_md.get_velocities()
        if existing_velocities is None:
            velocities = initialize_velocities(atoms_md, temperature=temperature)
        else:
            velocities = existing_velocities

    atoms_md.set_velocities(velocities)

    masses = atoms_md.get_masses()[:, None]
    dt = timestep 

    trajectory = [atoms_md.copy()]
    energies = [compute_energy(atoms_md, descriptor, gp_model)]

    if not run_loop or n_steps == 0:
        return np.asarray(trajectory), np.asarray(energies)

    forces = compute_forces_fd(atoms_md, descriptor, gp_model, epsilon=fd_epsilon)

    for _ in range(n_steps):
        velocities = atoms_md.get_velocities()
        accel = forces / masses

        velocities_half = velocities + 0.5 * dt * accel
        new_positions = atoms_md.get_positions() + dt * velocities_half
        atoms_md.set_positions(new_positions)

        new_forces = compute_forces_fd(atoms_md, descriptor, gp_model, epsilon=fd_epsilon)
        new_accel = new_forces / masses

        new_velocities = velocities_half + 0.5 * dt * new_accel

        if use_velocity_scaling_thermostat:
            new_velocities = velocity_scaling_thermostat(
                new_velocities,
                atoms_md.get_masses(),
                target_temperature=temperature,
            )

        atoms_md.set_velocities(new_velocities)

        forces = new_forces

        trajectory.append(atoms_md.copy())
        energies.append(compute_energy(atoms_md, descriptor, gp_model))

    return trajectory, np.asarray(energies)

