from abc import ABC, abstractmethod

import taichi as ti
from taichi.math import vec3

from .mesh import Mesh


@ti.data_oriented
class ParametricMesh(ABC, Mesh):
    def __init__(self) -> None: ...

    @abstractmethod
    @ti.func
    def kernel(self, u: ti.f32, v: ti.f32) -> vec3: ...

    @abstractmethod
    @ti.kernel
    def generate(
        self,
        u_res: ti.i32,
        v_res: ti.u32,
    ):
        for i, j in ti.ndrange(u_res, v_res):
            u = i / (u_res - 1)
            v = j / (v_res - 1)
            self.vertices[i, j] = self.kernel(u, v)
