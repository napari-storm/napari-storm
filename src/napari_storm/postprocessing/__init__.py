"""Post-processing algorithms. COMET is imported only when a run is requested."""

from .drift import DriftModel

__all__ = ["DriftModel"]
