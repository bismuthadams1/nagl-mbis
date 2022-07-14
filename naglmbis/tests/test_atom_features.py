import numpy as np

from naglmbis.features.atom import (
    AtomicPolarisability,
    HydrogenAtoms,
    LipinskiAcceptor,
    LipinskiDonor,
    PaulingElectronegativity,
    SandersonElectronegativity,
    vdWRadius,
)


def test_hydrogen_atoms(methanol):
    """
    Make sure we get the correct atom features for the hydrogen atom features.
    """
    feat = HydrogenAtoms()
    assert len(feat) == 5

    feats = feat(methanol).numpy()
    assert feats.shape == (6, 5)
    assert np.allclose(feats[0], np.array([0, 0, 0, 1, 0]))


def test_lipinski_donor(methanol):
    """Make sure Lipinski donors are correctly tagged."""

    ld = LipinskiDonor()
    assert len(ld) == 1
    feats = ld(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(feats, np.array([[0], [1], [0], [0], [0], [0]]))


def test_lipinski_acceptor(methanol):
    """Make sure lipinski acceptors are correctly tagged."""

    la = LipinskiAcceptor()
    assert len(la) == 1
    feats = la(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(feats, np.array([[0], [1], [0], [0], [0], [0]]))


def test_pauling(methanol):
    """Make sure the electronegativity of each atom is correctly assigned"""

    pe = PaulingElectronegativity()
    assert len(pe) == 1
    feats = pe(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(feats, np.array([[2.55], [3.44], [2.2], [2.2], [2.2], [2.2]]))


def test_sanderson(methanol):
    """Make sure the Sanderson electronegativity is correctly assigned"""

    se = SandersonElectronegativity()
    assert len(se) == 1
    feats = se(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(
        feats, np.array([[2.75], [3.65], [2.59], [2.59], [2.59], [2.59]])
    )


def test_vdw_radii(methanol):
    """Make sure the vdw radii is correctly assigned"""

    radii = vdWRadius()
    assert len(radii) == 1
    feats = radii(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(feats, np.array([[1.75], [1.4], [1.17], [1.17], [1.17], [1.17]]))


def test_polarisability(methanol):

    polar = AtomicPolarisability()
    assert len(polar) == 1
    feats = polar(methanol).numpy()
    assert feats.shape == (6, 1)
    assert np.allclose(feats, np.array([[1.76], [1.1], [0.67], [0.67], [0.67], [0.67]]))
