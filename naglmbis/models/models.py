from typing import Literal

import torch

from naglmbis.models.base_model import MBISGraphModel
from naglmbis.utils import get_model_weights

charge_weights = {
    "nagl-v1-mbis": {"checkpoint_path": "nagl-v1-mbis.ckpt"},
    "nagl-v1-mbis-dipole": {"checkpoint_path": "nagl-v1-mbis-dipole.ckpt"},
    "nagl-gas-charge-wb": {"checkpoint_path" : "nagl-gas-charge.ckpt"},
    "nagl-gas-charge-dipole-wb": {"checkpoint_path" : "nagl-gas-charge-dipole.ckpt"},
    "nagl-gas-charge-dipole-esp-wb-default":{"checkpoint_path":"nagl-gas-charge-dipole-esp.ckpt"},
    "nagl-water-charge-wb":  {"checkpoint_path" : "nagl-water-charge.ckpt"},
    "nagl-water-charge-dipole-wb":  {"checkpoint_path" : "nagl-water-charge-dipole.ckpt"},
    "nagl-water-charge-dipole-esp-wb-default":{"checkpoint_path":"nagl-water-charge-dipole-esp.ckpt"},
    "nagl-gas-esp-wb-2A": {"checkpoint_path":"nagl-gas-esp-2A.ckpt"},
    "nagl-gas-esp-wb-15A":{"checkpoint_path": "nagl-gas-esp-15A.ckpt"},
}

CHARGE_MODELS = Literal["nagl-v1-mbis-dipole",
                        "nagl-v1-mbis",
                        "nagl-gas-charge-wb",
                        "nagl-gas-charge-dipole-wb",
                        "nagl-gas-charge-dipole-esp-wb-default"
                        "nagl-water-charge-wb",
                        "nagl-water-charge-dipole-wb",
                        "nagl-water-charge-dipole-esp-wb-default",
                        "nagl-gas-esp-wb-2A",
                        "nagl-gas-esp-wb-15A",
                        ]


def load_charge_model(charge_model: CHARGE_MODELS) -> MBISGraphModel:
    """
    Load up one of the predefined charge models, this will load the weights and parameter settings.
    """
    weight_path = get_model_weights(
        model_type="charge", model_name=charge_weights[charge_model]["checkpoint_path"]
    )
    model_data = torch.load(weight_path, map_location=torch.device('cpu'))
    model = MBISGraphModel(**model_data["hyper_parameters"])
    model.load_state_dict(model_data["state_dict"])
    model.eval()
    return model


# def load_volume_model(volume_model: VOLUME_MODELS) -> MBISGraphModel:
#     """
#     Load one of the predefined volume models, this will load the weights and parameter settings.
#     """
#     weight_path = get_model_weights(
#         model_type="volume", model_name=volume_weights[volume_model]["path"]
#     )
#     return volume_weights[volume_model]["model"].load_from_checkpoint(weight_path)
