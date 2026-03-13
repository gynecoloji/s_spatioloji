"""Cell morphology metrics from polygon boundaries.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _fractal_dimension(coords: np.ndarray) -> float:
    """Box-counting fractal dimension of a 2D point set.

    Args:
        coords: Array of shape (N, 2) with boundary coordinates.

    Returns:
        Estimated fractal dimension (typically 1.0-1.5 for cell boundaries).
    """
    if len(coords) < 4:
        return 1.0

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normed = (coords - mins) / span

    sizes = []
    counts = []
    for k in range(1, 8):
        cell_size = 1.0 / (2**k)
        bins = set()
        for pt in normed:
            bx = int(pt[0] / cell_size)
            by = int(pt[1] / cell_size)
            bins.add((bx, by))
        if len(bins) > 0:
            sizes.append(cell_size)
            counts.append(len(bins))

    if len(sizes) < 2:
        return 1.0

    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts, dtype=float))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(max(1.0, coeffs[0]))


def cell_morphology(
    sj: s_spatioloji,
    output_key: str = "morphology",
    force: bool = True,
) -> str:
    """Compute 13 morphology metrics per cell from polygon boundaries.

    Metrics include area, perimeter, centroid, circularity, elongation,
    solidity, eccentricity, aspect ratio, fractal dimension, vertex count,
    convexity defects, and rectangularity.

    Args:
        sj: Dataset instance.
        output_key: Key to write morphology table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> cell_morphology(sj)
        'morphology'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    gdf = sj.boundaries.load()

    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        cell_id = row["cell_id"]

        # Basic
        area = geom.area
        perimeter = geom.length
        centroid = geom.centroid
        centroid_x = centroid.x
        centroid_y = centroid.y

        # Shape descriptors
        circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Minimum rotated rectangle for elongation and aspect_ratio
        mrr = geom.minimum_rotated_rectangle
        mrr_coords = list(mrr.exterior.coords)
        edge_lengths = []
        for k in range(4):
            dx = mrr_coords[k + 1][0] - mrr_coords[k][0]
            dy = mrr_coords[k + 1][1] - mrr_coords[k][1]
            edge_lengths.append(np.sqrt(dx**2 + dy**2))
        major = max(edge_lengths)
        minor = min(edge_lengths)
        elongation = 1.0 - (minor / major) if major > 0 else 0.0
        aspect_ratio = major / minor if minor > 0 else 1.0

        # Solidity
        convex_hull = geom.convex_hull
        hull_area = convex_hull.area
        solidity = area / hull_area if hull_area > 0 else 1.0

        # Eccentricity via OpenCV fitEllipse
        ext_coords = np.array(geom.exterior.coords[:-1], dtype=np.float32)
        if len(ext_coords) >= 5:
            contour = ext_coords.reshape(-1, 1, 2)
            (_, _), (ma, MA), _ = cv2.fitEllipse(contour)
            if MA > 0:
                ratio = min(ma, MA) / max(ma, MA)
                eccentricity = np.sqrt(1.0 - ratio**2)
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        # Boundary complexity
        fd = _fractal_dimension(ext_coords)
        vertex_count = len(ext_coords)
        convexity_defects = hull_area - area

        # Rectangularity
        bbox = geom.bounds  # (minx, miny, maxx, maxy)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        rectangularity = area / bbox_area if bbox_area > 0 else 0.0

        records.append(
            {
                "cell_id": cell_id,
                "area": area,
                "perimeter": perimeter,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "circularity": circularity,
                "elongation": elongation,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "aspect_ratio": aspect_ratio,
                "fractal_dimension": fd,
                "vertex_count": vertex_count,
                "convexity_defects": convexity_defects,
                "rectangularity": rectangularity,
            }
        )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
