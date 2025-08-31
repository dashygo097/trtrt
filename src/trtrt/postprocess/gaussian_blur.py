from typing import Dict

import taichi as ti
from taichi.math import vec3

from .post_processor import ProcessorCore


@ti.data_oriented
class GaussianBlur(ProcessorCore):
    def __init__(
        self,
        enabled: bool = True,
        radius: int = 2,
        weight: float = 0.1,
        sigma: float = 1.0,
    ) -> None:
        self.radius = ti.field(dtype=ti.i32, shape=())
        self.weight = ti.field(dtype=ti.f32, shape=())
        self.sigma = ti.field(dtype=ti.f32, shape=())

        self.radius[None] = radius
        self.weight[None] = weight
        self.sigma[None] = sigma

        self.params: Dict = {}
        super().__init__(enabled)
        self.update()

    def _name(self) -> str:
        return "Gaussian Blur"

    def update(self) -> None:
        super().update()
        # Update params
        self.params["radius"] = self.radius[None]
        self.params["weight"] = self.weight[None]
        self.params["sigma"] = self.sigma[None]

    def set_weight(self, weight: float) -> None:
        self.weight[None] = weight
        self.update()

    def set_radius(self, radius: int) -> None:
        self.radius[None] = radius
        self.update()

    def set_sigma(self, sigma: float) -> None:
        self.sigma[None] = sigma
        self.update()

    @ti.kernel
    def process(self):
        # TODO: Implement bloom effect
        for i, j in self.buffers:
            blur_sum = vec3(0.0)
            weight_sum = 0.0

            for dx in range(-self.radius[None], self.radius[None] + 1):
                x = ti.min(self.res[0] - 1, ti.max(0, i + dx))
                weight = ti.exp(
                    -dx
                    * dx
                    / (
                        2.0
                        * 2
                        * self.sigma[None]
                        * self.sigma[None]
                        * self.radius[None]
                    )
                )
                blur_sum += weight * self.buffers[x, j]
                weight_sum += weight

            if weight_sum > 0:
                self.temp_buffers[i, j] = blur_sum / weight_sum

            else:
                self.temp_buffers[i, j] = vec3(0.0)

        for i, j in self.buffers:
            blur_sum = vec3(0.0)
            weight_sum = 0.0

            for dy in range(-self.radius[None], self.radius[None] + 1):
                y = ti.min(self.res[1] - 1, ti.max(0, j + dy))
                weight = ti.exp(
                    -dy
                    * dy
                    / (
                        2.0
                        * 2
                        * self.sigma[None]
                        * self.sigma[None]
                        * self.radius[None]
                    )
                )
                blur_sum += weight * self.temp_buffers[i, y]
                weight_sum += weight

            if weight_sum > 0.0:
                self.temp_buffers[i, j] = blur_sum / weight_sum

            else:
                self.temp_buffers[i, j] = vec3(0.0)

        for i, j in self.buffers:
            # NOTE: This will make the whole img more and more blurred as monte carlo integration proceeds

            self.buffers[i, j] = self.temp_buffers[i, j] * self.weight[
                None
            ] + self.buffers[i, j] * (1 - self.weight[None])


class Bloom(GaussianBlur):
    def __init__(
        self,
        enabled: bool = True,
        radius: int = 2,
        weight: float = 0.1,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(enabled, radius, weight, sigma)

    def _name(self) -> str:
        return "Bloom"

    @ti.kernel
    def process(self): ...
