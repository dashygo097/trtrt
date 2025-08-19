from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GeometryData:
    vertices: np.ndarray
    indices: np.ndarray
    texture_coords: Optional[np.ndarray] = None
    coords_mapping: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.vertices.shape[1] != 3:
            raise ValueError("Vertices should have shape (N, 3)")
        if self.indices.shape[1] != 3:
            raise ValueError("Indices should have shape (M, 3)")

        max_index = np.max(self.indices)
        if max_index >= len(self.vertices):
            raise ValueError("Indices contain invalid vertex references")
