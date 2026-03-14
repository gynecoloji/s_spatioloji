"""Point-based spatial analysis.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "build_knn_graph": ("s_spatioloji.spatial.point.graph", "build_knn_graph"),
    "build_radius_graph": ("s_spatioloji.spatial.point.graph", "build_radius_graph"),
    "neighborhood_composition": ("s_spatioloji.spatial.point.neighborhoods", "neighborhood_composition"),
    "neighborhood_diversity": ("s_spatioloji.spatial.point.neighborhoods", "neighborhood_diversity"),
    "nth_order_neighbors": ("s_spatioloji.spatial.point.neighborhoods", "nth_order_neighbors"),
    "colocalization": ("s_spatioloji.spatial.point.patterns", "colocalization"),
    "morans_i": ("s_spatioloji.spatial.point.patterns", "morans_i"),
    "gearys_c": ("s_spatioloji.spatial.point.patterns", "gearys_c"),
    "clustering_coefficient": ("s_spatioloji.spatial.point.patterns", "clustering_coefficient"),
    "getis_ord_gi": ("s_spatioloji.spatial.point.patterns", "getis_ord_gi"),
    "ripley_k": ("s_spatioloji.spatial.point.ripley", "ripley_k"),
    "ripley_l": ("s_spatioloji.spatial.point.ripley", "ripley_l"),
    "ripley_g": ("s_spatioloji.spatial.point.ripley", "ripley_g"),
    "ripley_f": ("s_spatioloji.spatial.point.ripley", "ripley_f"),
    "permutation_test": ("s_spatioloji.spatial.point.statistics", "permutation_test"),
    "quadrat_density": ("s_spatioloji.spatial.point.statistics", "quadrat_density"),
    "clark_evans": ("s_spatioloji.spatial.point.statistics", "clark_evans"),
    "dclf_envelope": ("s_spatioloji.spatial.point.statistics", "dclf_envelope"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
