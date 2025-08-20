from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import taichi as ti
from taichi.math import vec3

from .mesh import Mesh


class ParametricMesh(ABC, Mesh):
    def __init__(self, **shape_params) -> None:
        super().__init__()
        self.shape_params: Dict[str, Any] = shape_params

    @abstractmethod
    def kernel(self, u: ti.i32, v: ti.f32) -> vec3: ...

    def generate(self, u_res: int, v_res: int) -> None:
        vertices: np.ndarray = np.zeros((u_res * v_res, 3), dtype=np.float32)
        indices: np.ndarray = np.zeros(
            ((u_res - 1) * (v_res - 1) * 2, 3), dtype=np.int32
        )

        for i in range(u_res):
            for j in range(v_res):
                u = i / (u_res - 1)
                v = j / (v_res - 1)
                vertices[i * u_res + j] = self.kernel(u, v)

        for i in range(u_res - 1):
            for j in range(v_res - 1):
                indices[i * (u_res - 1) + j] = [
                    i * v_res + j,
                    i * v_res + (j + 1),
                    (i + 1) * v_res + j,
                ]
                indices[(i * (u_res - 1) + j) + (u_res - 1) * (v_res - 1)] = [
                    i * v_res + (j + 1),
                    (i + 1) * v_res + (j + 1),
                    (i + 1) * v_res + j,
                ]

        self.load_geometry(vertices, indices)
