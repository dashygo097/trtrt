import taichi as ti
from taichi.math import vec3


@ti.dataclass
class GBuffer:
    depth: ti.f32
    pos: vec3
    normal: vec3
    albedo: vec3
