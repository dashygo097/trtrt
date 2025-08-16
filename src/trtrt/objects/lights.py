import taichi as ti
from taichi.math import vec3


@ti.dataclass
class Ray:
    origin: vec3
    dir: vec3

    @ti.func
    def at(self, t: ti.f32) -> vec3:
        return self.origin + t * self.dir


@ti.dataclass
class DirecLight:
    dir: vec3
    color: vec3
