from typing import Dict

import taichi as ti
from taichi.math import vec3

from ..objects import Ray
from .base import Renderer
from .sampler import Sampler


@ti.data_oriented
class Normal(Renderer):
    def __init__(
        self,
        sampler: Sampler = Sampler(),
        samples_per_pixel: int = 4,
    ) -> None:
        self.params: Dict = {}
        super().__init__(sampler, samples_per_pixel)
        self.update()

    def _name(self) -> str:
        return "Normal"

    def update(self) -> None:
        # Update params
        super().update()

    @ti.func
    def ray_color(self, scene, ray: Ray):  # pyright: ignore
        color_buffer = vec3(0.0)
        hitinfo = scene.intersect(ray)
        color_buffer = ti.abs(hitinfo.normal + 1) * 0.5
        return color_buffer
