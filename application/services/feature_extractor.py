from collections import deque
import math
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

ML_FEATURE_NAMES = [
    "width", "height", "floor_ratio", "n_regions", "largest_region_ratio",
    "edge_density", "row_entropy", "col_entropy", "local_variance",
    "perimeter_area_ratio",
]

FEATURE_NAMES = ML_FEATURE_NAMES.copy()

def _binary_entropy(values):
    arr = np.clip(np.asarray(values, dtype=float), 1e-9, 1 - 1e-9)
    return -(arr * np.log2(arr) + (1 - arr) * np.log2(1 - arr))


def _regions(grid):
    h, w = grid.shape
    visited = set()
    sizes = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 1 or (x, y) in visited:
                continue
            q = deque([(x, y)])
            visited.add((x, y))
            size = 0
            while q:
                cx, cy = q.popleft()
                size += 1
                for nx, ny in ((cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)):
                    if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
            sizes.append(size)
    return sizes


def extract_ml_feature_dict(grid):
    arr = np.asarray(grid, dtype=np.uint8)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(
            "El mapa debe ser una matriz 2D no vacía"
        )

    height, width = arr.shape

    # Proporcion de celdas transitables
    floor_ratio = float(arr.mean())
    total_floor = max(int(arr.sum()), 1)

    # Regiones conectadas
    sizes = _regions(arr)
    n_regions = len(sizes)
    largest = max(sizes, default=0)

    largest_region_ratio = largest / total_floor

    # Diferencias entre celdas adyacentes
    horizontal_changes = (
        arr[:, 1:] != arr[:, :-1]
        if width > 1
        else np.empty((height, 0), dtype=bool)
    )

    vertical_changes = (
        arr[1:, :] != arr[:-1, :]
        if height > 1
        else np.empty((0, width), dtype=bool)
    )

    horizontal = (
        float(horizontal_changes.mean())
        if width > 1
        else 0.0
    )

    vertical = (
        float(vertical_changes.mean())
        if height > 1
        else 0.0
    )

    edge_density = float(
        (horizontal + vertical) / 2
    )

    # Entropia
    row_entropy = float(
        _binary_entropy(arr.mean(axis=1)).mean()
    )

    col_entropy = float(
        _binary_entropy(arr.mean(axis=0)).mean()
    )

    # ==========================================
    # HU-78: Varianza local vectorizada
    # ==========================================
    padded = np.pad(
        arr.astype(float),
        1,
        mode="edge"
    )

    windows = sliding_window_view(
        padded,
        window_shape=(3, 3)
    )

    local_variance = float(
        np.var(windows, axis=(-2, -1)).mean()
    )

    # Perimetro
    internal_edges = int(
        horizontal_changes.sum()
        + vertical_changes.sum()
    )

    border_open = int(
        arr[0, :].sum()
        + arr[-1, :].sum()
        + arr[:, 0].sum()
        + arr[:, -1].sum()
    )

    perimeter_area_ratio = float(
        (internal_edges + border_open) / total_floor
    )

    return {
        "width": int(width),
        "height": int(height),
        "floor_ratio": floor_ratio,
        "n_regions": int(n_regions),
        "largest_region_ratio": float(
            largest_region_ratio
        ),
        "edge_density": edge_density,
        "row_entropy": row_entropy,
        "col_entropy": col_entropy,
        "local_variance": local_variance,
        "perimeter_area_ratio": perimeter_area_ratio,
    }


def extract_feature_dict(grid, ml_features=None):
    """
    Compatibilidad:
    métricas antiguas + métricas estructurales
    del experimento final.

    HU-78:
    Permite reutilizar características ML
    previamente calculadas.
    """

    ml = (
        ml_features
        if ml_features is not None
        else extract_ml_feature_dict(grid)
    )

    accessibility = ml["floor_ratio"]
    obstacle_density = 1.0 - accessibility

    return {
        **ml,
        "connectivity": ml["largest_region_ratio"],
        "accessibility": accessibility,
        "obstacle_density": obstacle_density,
        "connected_components": ml["n_regions"],
        "largest_open_area": (
            ml["largest_region_ratio"] * accessibility
        ),
        "border_obstacle_ratio": 0.0,
        "balance": 1.0 - abs(accessibility - 0.65),
    }


def extract_features(grid):
    f = extract_ml_feature_dict(grid)
    return [f[name] for name in ML_FEATURE_NAMES]
