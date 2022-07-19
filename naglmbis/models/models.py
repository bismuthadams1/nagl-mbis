from typing import Literal

from nagl.features import (
    AtomConnectivity,
    AtomFormalCharge,
    AtomicElement,
    AtomIsAromatic,
)

from naglmbis.features.atom import (
    AtomicMass,
    AtomInRingOfSize,
    ExplicitValence,
    Hybridization,
    TotalDegree,
    TotalValence,
)
from naglmbis.models.base_model import MBISGraphModel
from naglmbis.utils import get_model_weights


class MBISGraphModelV1(MBISGraphModel):
    """
    The first version of the model with the basic set of features.
    """

    def features(self):
        atom_features = [
            AtomicElement(["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]),
            AtomConnectivity(),
            AtomInRingOfSize(3),
            AtomInRingOfSize(4),
            AtomInRingOfSize(5),
            AtomInRingOfSize(6),
        ]
        bond_features = []
        return atom_features, bond_features


class EspalomaModel(MBISGraphModel):
    """Try and recreate the espaloma model"""

    def features(self):
        atom_features = [
            AtomicElement(["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]),
            TotalDegree(),
            TotalValence(),
            ExplicitValence(),
            AtomFormalCharge(),
            AtomIsAromatic(),
            AtomicMass(),
            AtomInRingOfSize(3),
            AtomInRingOfSize(4),
            AtomInRingOfSize(5),
            AtomInRingOfSize(6),
            AtomInRingOfSize(7),
            AtomInRingOfSize(8),
            Hybridization(),
        ]
        bond_features = []
        return atom_features, bond_features


charge_weights = {1: {"path": "mbis_charges_v1.ckpt", "model": MBISGraphModelV1}}
volume_weights = {1: {"path": "mbis_volumes_v1.ckpt", "model": MBISGraphModelV1}}
CHARGE_MODELS = Literal[1]
VOLUME_MODELS = Literal[1]


def load_charge_model(charge_model: CHARGE_MODELS) -> MBISGraphModel:
    """
    Load up one of the predefined charge models, this will load the weights and parameter settings.
    """
    weight_path = get_model_weights(
        model_type="charge", model_name=charge_weights[charge_model]["path"]
    )
    return charge_weights[charge_model]["model"].load_from_checkpoint(weight_path)


def load_volume_model(volume_model: VOLUME_MODELS) -> MBISGraphModel:
    """
    Load one of the predefined volume models, this will load the weights and parameter settings.
    """
    weight_path = get_model_weights(
        model_type="volume", model_name=volume_weights[volume_model]["path"]
    )
    return volume_weights[volume_model]["model"].load_from_checkpoint(weight_path)
