from typing import Dict

import taichi as ti
from taichi.math import vec3

from ..objects import Ray
from .base import Renderer
from .sampler import Sampler, UniformSampler


@ti.data_oriented
class ZBuffer(Renderer):
    def __init__(
        self,
        sampler: Sampler = UniformSampler(),
        samples_per_pixel: int = 4,
        alpha: float = 20.0,
    ) -> None:
        self.alpha = ti.field(dtype=ti.f32, shape=())

        self.alpha[None] = alpha

        self.params: Dict = {}
        super().__init__(sampler, samples_per_pixel)
        self.update()

    def set_alpha(self, alpha: float) -> None:
        self.alpha[None] = alpha
        self.update()

    def update(self):
        super().update()
        self.params["alpha"] = self.alpha[None]

    @ti.func
    def ray_color(self, scene, ray: Ray):
        color_buffer = vec3(0.0)
        hitinfo = scene.intersect(ray)
        color_buffer = vec3(ti.tanh(hitinfo.time / self.alpha[None]) * 2 / ti.math.pi)
        return color_buffer
