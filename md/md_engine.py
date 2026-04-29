from __future__ import annotations

from typing import List

from ase import Atoms

from md.dynamics import run_velocity_verlet_md


class MDEngine:
    """Wrapper around MD integrators that returns ASE Atoms trajectory."""

    def __init__(
        self,
        mode: str = "verlet",
        timestep: float = 0.05,
        temperature: float = 300.0,
        use_thermostat: bool = False,
    ):
        self.mode = mode
        self.timestep = timestep
        self.temperature = temperature
        self.use_thermostat = use_thermostat

    def run(
        self,
        atoms: Atoms,
        descriptor,
        gp_model,
        n_steps: int = 20,
    ) -> List[Atoms]:

        if self.mode == "verlet":
            return self._run_verlet(atoms, descriptor, gp_model, n_steps)

        else:
            raise ValueError(f"Unknown MD mode: {self.mode}")

    # =========================
    # Verlet implementation
    # =========================
    def _run_verlet(
        self,
        atoms: Atoms,
        descriptor,
        gp_model,
        n_steps: int,
    ) -> List[Atoms]:

        trajectory, _ = run_velocity_verlet_md(
            atoms,
            descriptor,
            gp_model,
            timestep=self.timestep,
            n_steps=n_steps,
            temperature=self.temperature,
            use_velocity_scaling_thermostat=self.use_thermostat,
        )

        return trajectory
