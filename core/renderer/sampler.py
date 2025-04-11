import taichi as ti
from taichi.math import vec3

from ..utils.const import EPSILON
from .utils import reflect


@ti.data_oriented
class Sampler:
    def __init__(self) -> None:
        pass

    def _name(self) -> str:
        return self.__class__.__name__

    @ti.func
    def hemispherical_sample(self, n):
        u, v = ti.random(), ti.random()
        theta, phi = ti.acos(ti.sqrt(u)), v * 2 * ti.math.pi

        x = ti.sin(theta) * ti.cos(phi)
        y = ti.sin(theta) * ti.sin(phi)
        z = ti.cos(theta)
        vec = vec3(x, y, z)

        tangent = vec3(0.0)
        if ti.abs(n[0]) > ti.abs(n[1]):
            tangent = ti.Vector([n[2], 0.0, -n[0]]).normalized()
        else:
            tangent = ti.Vector([0.0, n[2], -n[1]]).normalized()

        bitangent = n.cross(tangent)

        result = vec[0] * tangent + vec[1] * bitangent + vec[2] * n

        return (result * ti.cos(theta) / ti.math.pi).normalized()

    @ti.func
    def sample_cone(self, dir, angle):
        z_axis = dir

        temp = vec3(1.0, 0.0, 0.0)
        if ti.abs(z_axis.dot(temp)) > 1 - EPSILON:
            temp = vec3(0.0, 1.0, 0.0)

        x_axis = temp.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()

        r = ti.sqrt(ti.random()) * ti.tan(angle / 180 * ti.math.pi)
        theta = 2.0 * ti.math.pi * ti.random()

        x = r * ti.cos(theta)
        y = r * ti.sin(theta)

        return (z_axis + x * x_axis + y * y_axis).normalized()

    @ti.func
    def ggx_sample(self, v, n, alpha):
        u = ti.random()
        _v = ti.random()

        alpha2 = alpha * alpha

        phi = 2.0 * ti.math.pi * u

        cos_theta = ti.sqrt((1.0 - _v) / (1.0 + (alpha2 - 1.0) * _v))
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        h_tangent = vec3(sin_theta * ti.cos(phi), cos_theta, sin_theta * ti.sin(phi))

        up = vec3(0.0, 1.0, 0.0)

        if abs(n[1]) > 0.999:
            up = vec3(1.0, 0.0, 0.0)

        tangent = up.cross(n).normalized()
        bitangent = n.cross(tangent)

        world_h = tangent * h_tangent[0] + n * h_tangent[1] + bitangent * h_tangent[2]
        world_h = world_h.normalized()

        l = reflect(-v, world_h)

        return l.normalized()
