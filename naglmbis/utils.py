import os

from pkg_resources import resource_filename
from typing_extensions import Literal


def get_model_weights(model_type: Literal["charge", "volume"], model_name: str) -> str:
    """
    Get the model weights from the naglmbis package.

    """

    fn = resource_filename(
        "naglmbis", os.path.join("data", "models", model_type, model_name)
    )
    if not os.path.exists(fn):
        raise ValueError(
            f"{model_name} does not exist. If you have just added it, you'll need to re-install."
        )
    return fn
