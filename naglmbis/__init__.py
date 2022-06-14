"""
naglmbis
Models built with NAGL to predict MBIS properties.
"""

from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
