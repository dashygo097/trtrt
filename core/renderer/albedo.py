from typing import Dict

import taichi as ti

from ..objects import Ray
from .base import Renderer
from .sampler import Sampler


@ti.data_oriented
class Albedo(Renderer):
    def __init__(
        self,
        sampler: Sampler = Sampler(),
        samples_per_pixel: int = 4,
    ) -> None:
        self.params: Dict = {}
        super().__init__(sampler, samples_per_pixel)
        self.update()

    def _name(self) -> str:
        return "Albedo"

    def update(self) -> None:
        # Update params
        super().update()

    @ti.func
    def ray_color(self, scene, ray: Ray):  # pyright: ignore
        hitinfo = scene.intersect(ray)
        return hitinfo.albedo
