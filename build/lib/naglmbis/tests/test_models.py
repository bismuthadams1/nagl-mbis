import torch

from naglmbis.models import MBISGraphModelV1, load_charge_model, load_volume_model


def test_charge_model_v1(methanol):
    """
    Test loading the charge model and computing the MBIS charges.
    """
    charge_model = load_charge_model(charge_model="nagl-v1")
    charges = charge_model.compute_properties(molecule=methanol)[
        "mbis-charges"
    ].detach()
    ref = torch.Tensor([[0.0847], [-0.6714], [0.0494], [0.0494], [0.0494], [0.4384]])
    assert torch.allclose(charges, ref, atol=1e-4)


def test_volume_model_v1(methanol):
    """
    Test loading the volume model and computing the MBIS volumes
    """
    volume_model = load_volume_model(volume_model="nagl-v1")
    volumes = volume_model.compute_properties(molecule=methanol)[
        "mbis-volumes"
    ].detach()
    ref = torch.Tensor([[29.6985], [25.3335], [3.0224], [3.0224], [3.0224], [1.0341]])
    assert torch.allclose(volumes, ref, atol=1e-4)


def test_activation():
    """
    Test building a model with a new activation function
    """
    model = MBISGraphModelV1(
        n_gcn_hidden_features=128,
        n_gcn_layers=3,
        n_mbis_hidden_features=128,
        n_mbis_layers=2,
        readout_modules="charge",
        learning_rate=1e-4,
        gcn_activation=None,
        readout_activation="Sigmoid",
    )

    assert model.readout_activation == "Sigmoid"
    assert isinstance(
        model.readout_modules["mbis-charges"].readout_layers[1], torch.nn.Sigmoid
    )
