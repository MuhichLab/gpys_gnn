def test_descriptor_shapes():
    from ase.build import molecule

    mols = [
        molecule("CH4"),
        molecule("CH3OH"),
        molecule("C2H5OH"),
    ]

    shapes = []
    for m in mols:
        X = descriptor.create(m)
        shapes.append(X.shape[1])

    assert len(set(shapes)) == 1, "Descriptor feature size mismatch"
