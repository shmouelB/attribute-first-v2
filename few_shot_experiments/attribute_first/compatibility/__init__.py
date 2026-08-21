"""Compatibility boundary for historical experiment identifiers."""

from .legacy_names import LegacyNameResolver, UnknownLegacyNameError

__all__ = ["LegacyNameResolver", "UnknownLegacyNameError"]
