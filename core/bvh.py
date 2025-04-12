import taichi as ti
from taichi.math import vec3

from .records import BVHHitInfo


@ti.dataclass
class AABB:
    min: vec3
    max: vec3


@ti.dataclass
class BVHNode:
    aabb: AABB
    left_id: ti.i32
    right_id: ti.i32
    obj_id: ti.i32
