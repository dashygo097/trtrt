import taichi as ti
from taichi.math import vec3

from ..parametric import ParametricMesh


class ParametricTorus(ParametricMesh):
    def __init__(self, **shape_params) -> None:
        super().__init__()
        self.shape_params = shape_params

    def kernel(self, u: ti.i32, v: ti.f32) -> vec3:
        R = self.shape_params.get("R", 1.0)
        r = self.shape_params.get("r", 0.5)

        _u = 2 * ti.math.pi * u
        _v = 2 * ti.math.pi * v
        x = (R + r * ti.cos(_v)) * ti.cos(_u)
        y = (R + r * ti.cos(_v)) * ti.sin(_u)
        z = r * ti.sin(_v)
        return vec3(x, y, z)
