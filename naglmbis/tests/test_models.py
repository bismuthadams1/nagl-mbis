import torch

from naglmbis.models import load_charge_model


def test_charge_model_v1_dipoles(methanol):
    """
    Test loading the charge model and computing the MBIS charges with a model co-trained to dipoles.
    """
    charge_model = load_charge_model(charge_model="nagl-v1-mbis-dipole")
    charges = charge_model.compute_properties(molecule=methanol)[
        "mbis-charges"
    ].detach()
    ref = torch.Tensor([[0.0618], [-0.6490], [0.0509], [0.0509], [0.0509], [0.4347]])
    assert torch.allclose(charges, ref, atol=1e-4)


def test_charge_model_v1_mbis(methanol):
    """Test computing the charges with the model trained to only mbis charges."""
    charge_model = load_charge_model(charge_model="nagl-v1-mbis")
    charges = charge_model.compute_properties(molecule=methanol)[
        "mbis-charges"
    ].detach()
    ref = torch.Tensor([[0.0835], [-0.6821], [0.0491], [0.0491], [0.0491], [0.4515]])
    assert torch.allclose(charges, ref, atol=1e-4)
