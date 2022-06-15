import pytest
from openff.toolkit.topology import Molecule


@pytest.fixture()
def methanol():
    """
    Make methanol with a specific atom ordering.
    """
    methanol = Molecule.from_mapped_smiles("[H:3][C:1]([H:4])([H:5])[O:2][H:6]")
    methanol.generate_conformers(n_conformers=1)
    return methanol


@pytest.fixture()
def water():
    """Make an OpenFF molecule of water"""
    water = Molecule.from_mapped_smiles("[H:2][O:1][H:3]")
    water.generate_conformers(n_conformers=1)
    return water
