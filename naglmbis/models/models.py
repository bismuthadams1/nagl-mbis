from typing import Literal

import torch

from naglmbis.models.base_model import MBISGraphModel
from naglmbis.utils import get_model_weights

charge_weights = {
    "nagl-v1-mbis": {"checkpoint_path": "nagl-v1-mbis.ckpt"},
    "nagl-v1-mbis-dipole": {"checkpoint_path": "nagl-v1-mbis-dipole.ckpt"},
}
# volume_weights = {
#     "nagl-v1": {"path": "mbis_volumes_v1.ckpt", "model": MBISGraphModel}
# }
CHARGE_MODELS = Literal["nagl-v1-mbis-dipole", "nagl-v1-mbis"]
# VOLUME_MODELS = Literal["nagl-v1"]


def load_charge_model(charge_model: CHARGE_MODELS) -> MBISGraphModel:
    """
    Load up one of the predefined charge models, this will load the weights and parameter settings.
    """
    weight_path = get_model_weights(
        model_type="charge", model_name=charge_weights[charge_model]["checkpoint_path"]
    )
    model_data = torch.load(weight_path)
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
