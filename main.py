from __future__ import annotations

import logging

from ase.build import bulk
from ase.calculators.emt import EMT

from active_learning.gp_cycle import run_gp_active_learning_cycle
from dft.run_dft import run_dft
from md.md_engine import MDEngine
from models.descriptors import SOAPDescriptor
from models.gp_model import AtomicGaussianProcess

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_initial_structures() -> list:
	structures = []



	return structures


def main(n_cycles: int = 3, n_select : int = 2, md_steps: int = 10) -> None: 
	#initialize components
	descriptor=SOAPDescriptor(
		species  = [],
		cutoff_radius = 5.0,
		n_radial = 8,
		n_angular = 6,
	)
	gp_model = GPModel()
	md_engine=MDEngin()
	dft_calculator=EMT() #placeholder

	# initialize data set
	dataset_structures = build_initial_structures()
	dataset_energies = []
	for atoms in dataset_structures:
    		E, _ = run_dft(atoms, dft_calculator)
    		dataset_energies.append(E)


	X_list = [descriptor.create(a) for a in dataset_structures]
	gp_model.fit(X_list, dataset_energies)
	logger.info("Initilized data set with %d structures", len(dataset_structures))

	#active learning loop
	for cycle in range(n_cycles):
		logger.info("Stareting active learning cycles %d", cycle)

		#Run MD using current surrogate workflow
		see_structure = dataset_structures[-1]
		trajectory_structures = md_engine.run(
			seed_structure, 
			calculator=dft_calculator,
			n_steps=md_steps,
		)

		# Select structures baesd on GP uncertainty
		results = run_gp_active_learning_cycle(
			candidate_structures=trajectory_structures ,
			gp_model=gp_model,
			descriptor= descriptor, 
			n_select = n_select,
		)

		selected_structures = results["selected_structures"]
		selected_energies = run_dft(selected_structures,dft_calculator)

		#Append data and retrain
		dataset_structures.extend(selected_structures)
		dataset_energies.extent(selected_energies)
		gp_model.fit(dataset_structures, dataset_energies, descriptor)

		logger.info(
			"Cycle %d complete | selected %d | max_uncertainty=%.6f | dataset_size=%d",
			cycle,
			len(selected_structures),
			results["max_uncertainty"],
			len(dataset_structures),
		)

if __name__ =="__main__":
	main()

