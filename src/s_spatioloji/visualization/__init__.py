"""Visualization functions for s_spatioloji.

All plotting functions accept an ``s_spatioloji`` object, return
``matplotlib.axes.Axes`` (or ``Figure`` for multi-panel plots),
and auto-save to ``{dataset_root}/figures/``.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "scatter": ("s_spatioloji.visualization.embedding", "scatter"),
    "spatial_scatter": ("s_spatioloji.visualization.spatial", "spatial_scatter"),
    "spatial_polygons": ("s_spatioloji.visualization.spatial", "spatial_polygons"),
    "spatial_expression": ("s_spatioloji.visualization.spatial", "spatial_expression"),
    "heatmap": ("s_spatioloji.visualization.expression", "heatmap"),
    "violin": ("s_spatioloji.visualization.expression", "violin"),
    "dotplot": ("s_spatioloji.visualization.expression", "dotplot"),
    "ripley_plot": ("s_spatioloji.visualization.analysis", "ripley_plot"),
    "colocalization_heatmap": ("s_spatioloji.visualization.analysis", "colocalization_heatmap"),
    "neighborhood_bar": ("s_spatioloji.visualization.analysis", "neighborhood_bar"),
    "envelope_plot": ("s_spatioloji.visualization.analysis", "envelope_plot"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
