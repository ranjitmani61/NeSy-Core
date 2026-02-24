"""
nesy/version.py
===============
Single source of truth for NeSy-Core version information.

PEP 440 compliant versioning: MAJOR.MINOR.PATCH[-PRE]

Import this module for programmatic version access:
    from nesy.version import __version__, VERSION_INFO
"""

from __future__ import annotations

from typing import NamedTuple


class VersionInfo(NamedTuple):
    """Structured version information."""
    major: int
    minor: int
    patch: int
    pre_release: str = ""

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            return f"{base}-{self.pre_release}"
        return base


VERSION_INFO = VersionInfo(major=0, minor=1, patch=1)

__version__: str = str(VERSION_INFO)

# Minimum Python version required
PYTHON_REQUIRES = ">=3.9"

# Framework metadata
FRAMEWORK_NAME = "NeSy-Core"
FRAMEWORK_DESCRIPTION = "Neuro-Symbolic AI Reasoning Framework"
AUTHOR = "NeSy-Core Contributors"
LICENSE = "MIT"
