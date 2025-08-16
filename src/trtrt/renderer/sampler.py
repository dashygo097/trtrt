import importlib.resources as resources
from abc import ABC, abstractmethod

import numpy as np
import taichi as ti
from PIL import Image
from taichi.math import vec3

from .. import assets
from ..utils import EPSILON
from .utils import reflect


@ti.data_oriented
class Sampler(ABC):
    def __init__(self) -> None: ...

    def _name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    @ti.func
    def kernel(self, _u: ti.f32, _v: ti.f32) -> ti.f32:
        return 0.0

    @ti.func
    def hemispherical_sample(self, n: vec3, _u, _v) -> vec3:
        u, v = self.kernel(_u, _v), self.kernel(_u, _v)
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
    def sample_cone(self, dir: vec3, angle: ti.f32, _u: ti.f32, _v: ti.f32) -> vec3:
        z_axis = dir

        temp = vec3(1.0, 0.0, 0.0)
        if ti.abs(z_axis.dot(temp)) > 1 - EPSILON:
            temp = vec3(0.0, 1.0, 0.0)

        x_axis = temp.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()

        r = ti.sqrt(self.kernel(_u, _v)) * ti.tan(angle / 180 * ti.math.pi)
        theta = 2.0 * ti.math.pi * self.kernel(_u, _v)

        x = r * ti.cos(theta)
        y = r * ti.sin(theta)

        return (z_axis + x * x_axis + y * y_axis).normalized()

    @ti.func
    def ggx_sample(
        self, view: vec3, n: vec3, alpha: ti.f32, _u: ti.f32, _v: ti.f32
    ) -> vec3:
        u, v = self.kernel(_u, _v), self.kernel(_u, _v)

        alpha2 = alpha * alpha

        phi = 2.0 * ti.math.pi * u

        cos_theta = ti.sqrt((1.0 - v) / (1.0 + (alpha2 - 1.0) * v))
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        h_tangent = vec3(sin_theta * ti.cos(phi), cos_theta, sin_theta * ti.sin(phi))

        up = vec3(0.0, 1.0, 0.0)

        if abs(n[1]) > 0.999:
            up = vec3(1.0, 0.0, 0.0)

        tangent = up.cross(n).normalized()
        bitangent = n.cross(tangent)

        world_h = tangent * h_tangent[0] + n * h_tangent[1] + bitangent * h_tangent[2]
        world_h = world_h.normalized()

        l = reflect(-view, world_h)

        return l.normalized()


@ti.data_oriented
class UniformSampler(Sampler):
    def __init__(self) -> None:
        super().__init__()

    @ti.func
    def kernel(self, _u: ti.f32, _v: ti.f32) -> ti.f32:
        return ti.random(ti.f32)


@ti.data_oriented
class BlueNoiseSampler(Sampler):
    def __init__(self) -> None:
        super().__init__()
        with resources.open_binary(assets, "BlueNoise470.png") as f:
            bluenoise = Image.open(f)
            bluenoise = bluenoise.convert("L")
            bluenoise = np.array(bluenoise).astype(np.float32) / 255.0

            f.close()

        self.blue_noise = ti.field(dtype=ti.f32, shape=bluenoise.shape)
        self.blue_noise.from_numpy(bluenoise)

    @ti.func
    def kernel(self, _u: ti.f32, _v: ti.f32) -> ti.f32:
        texture_width, texture_height = self.blue_noise.shape
        jitter = 0.1
        _u = _u + ti.random(ti.f32) * jitter
        _v = _v + ti.random(ti.f32) * jitter
        tx = ti.i32(_u * texture_width) % texture_width
        ty = ti.i32(_v * texture_height) % texture_height

        result = self.blue_noise[tx, ty]

        return result
