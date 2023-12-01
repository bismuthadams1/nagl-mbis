"""
naglmbis
Models built with NAGL to predict MBIS properties.
"""
from . import _version

__version__ = _version.get_versions()["version"]
# make sure all custom features are loaded
import naglmbis.features

__all__ = [naglmbis.features]
