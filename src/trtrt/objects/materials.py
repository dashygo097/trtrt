import taichi as ti
from taichi.math import vec3


@ti.dataclass
class PBRMaterial:
    albedo: vec3
    metallic: ti.f32
    roughness: ti.f32
    emission: vec3


@ti.dataclass
class GlassMaterial:
    albedo: vec3
    refraction: ti.f32
