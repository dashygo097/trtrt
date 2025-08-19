import taichi as ti
from taichi.math import vec3


@ti.dataclass
class HitInfo:
    is_hit: ti.u1
    time: ti.f32
    pos: vec3
    normal: vec3
    front: ti.u1
    tag: ti.u32
    u: ti.f32
    v: ti.f32

    albedo: vec3
    metallic: ti.f32
    roughness: ti.f32
    emission: vec3
    refraction: ti.f32


@ti.dataclass
class BVHHitInfo:
    is_hit: ti.u1
    tmin: ti.f32
    tmax: ti.f32
