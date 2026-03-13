"""Polygon-based spatial analysis.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "build_contact_graph": ("s_spatioloji.spatial.polygon.graph", "build_contact_graph"),
    "cell_morphology": ("s_spatioloji.spatial.polygon.morphology", "cell_morphology"),
    "neighborhood_composition": ("s_spatioloji.spatial.polygon.neighborhoods", "neighborhood_composition"),
    "neighborhood_diversity": ("s_spatioloji.spatial.polygon.neighborhoods", "neighborhood_diversity"),
    "nth_order_neighbors": ("s_spatioloji.spatial.polygon.neighborhoods", "nth_order_neighbors"),
    "colocalization": ("s_spatioloji.spatial.polygon.patterns", "colocalization"),
    "morans_i": ("s_spatioloji.spatial.polygon.patterns", "morans_i"),
    "gearys_c": ("s_spatioloji.spatial.polygon.patterns", "gearys_c"),
    "clustering_coefficient": ("s_spatioloji.spatial.polygon.patterns", "clustering_coefficient"),
    "border_enrichment": ("s_spatioloji.spatial.polygon.patterns", "border_enrichment"),
    "permutation_test": ("s_spatioloji.spatial.polygon.statistics", "permutation_test"),
    "quadrat_density": ("s_spatioloji.spatial.polygon.statistics", "quadrat_density"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
