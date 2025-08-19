import taichi as ti

from ..utils.const import TMIN
from .entities import Sphere, Triangle


def init4bbox(obj):
    if isinstance(obj, Triangle):
        init_triangle(obj)
    elif isinstance(obj, Sphere):
        init_sphere(obj)
    else:
        raise ValueError("Unknown object type")


# NOTE: LEAVE THIS FOR TEMPORARY USE
def init_triangle(triangle: Triangle) -> None:
    triangle.bbox.min.x = ti.min(triangle.v0.x, triangle.v1.x, triangle.v2.x) - TMIN
    triangle.bbox.max.x = ti.max(triangle.v0.x, triangle.v1.x, triangle.v2.x) + TMIN
    triangle.bbox.min.y = ti.min(triangle.v0.y, triangle.v1.y, triangle.v2.y) - TMIN
    triangle.bbox.max.y = ti.max(triangle.v0.y, triangle.v1.y, triangle.v2.y) + TMIN
    triangle.bbox.min.z = ti.min(triangle.v0.z, triangle.v1.z, triangle.v2.z) - TMIN
    triangle.bbox.max.z = ti.max(triangle.v0.z, triangle.v1.z, triangle.v2.z) + TMIN


def init_sphere(sphere: Sphere) -> None:
    sphere.bbox.min.x = sphere.center.x - sphere.radius
    sphere.bbox.max.x = sphere.center.x + sphere.radius
    sphere.bbox.min.y = sphere.center.y - sphere.radius
    sphere.bbox.max.y = sphere.center.y + sphere.radius
    sphere.bbox.min.z = sphere.center.z - sphere.radius
    sphere.bbox.max.z = sphere.center.z + sphere.radius
