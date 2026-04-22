def compute_forces_fd(atoms, descriptor, gp, eps=1e-3):
    """
    Compute forces via finite differences on GP energy.

    atoms: ASE Atoms
    descriptor: SOAPDescriptor
    gp: AtomicGaussianProcess
    """

    positions = atoms.get_positions()
    forces = np.zeros_like(positions)

    for i in range(len(atoms)):
        for d in range(3):

            atoms_plus = atoms.copy()
            atoms_minus = atoms.copy()

            pos = positions.copy()

            pos[i, d] += eps
            atoms_plus.set_positions(pos)

            pos[i, d] -= 2 * eps
            atoms_minus.set_positions(pos)

            X_plus = descriptor.create(atoms_plus)
            X_minus = descriptor.create(atoms_minus)

            E_plus = gp.predict_mean([X_plus])[0]
            E_minus = gp.predict_mean([X_minus])[0]

            forces[i, d] = -(E_plus - E_minus) / (2 * eps)

    return forces
