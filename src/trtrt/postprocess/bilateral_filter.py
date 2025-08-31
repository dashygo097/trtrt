from typing import Dict

import taichi as ti
from taichi.math import vec3

from .post_processor import ProcessorCore


@ti.data_oriented
class BilateralFilter(ProcessorCore):
    def __init__(
        self,
        enabled: bool = False,
        radius: int = 2,
        weight: float = 0.1,
        sigma_d: float = 1.0,
        sigma_r: float = 1.0,
    ) -> None:
        self.radius = ti.field(dtype=ti.i32, shape=())
        self.weight = ti.field(dtype=ti.f32, shape=())
        self.sigma_d = ti.field(dtype=ti.f32, shape=())
        self.sigma_r = ti.field(dtype=ti.f32, shape=())

        self.radius[None] = radius
        self.weight[None] = weight
        self.sigma_d[None] = sigma_d
        self.sigma_r[None] = sigma_r

        self.params: Dict = {}
        super().__init__(enabled=enabled)

    def _name(self) -> str:
        return "Bilateral Filter"

    def set_radius(self, radius: int) -> None:
        self.radius[None] = radius
        self.update()

    def set_weight(self, weight: float) -> None:
        self.weight[None] = weight
        self.update()

    def set_sigma_d(self, sigma_d: float) -> None:
        self.sigma_d[None] = sigma_d
        self.update()

    def set_sigma_r(self, sigma_r: float) -> None:
        self.sigma_r[None] = sigma_r
        self.update()

    def update(self) -> None:
        super().update()
        self.params["radius"] = self.radius[None]
        self.params["weight"] = self.weight[None]
        self.params["sigma_d"] = self.sigma_d[None]
        self.params["sigma_r"] = self.sigma_r[None]

    @ti.kernel
    def process(self):
        for i, j in self.buffers:
            color = self.buffers[i, j]
            filter_sum = vec3(0.0)
            weight_sum = 0.0

            for dx in range(-self.radius[None], self.radius[None] + 1):
                x = ti.min(self.res[0] - 1, ti.max(0, i + dx))
                for dy in range(-self.radius[None], self.radius[None] + 1):
                    y = ti.min(self.res[1] - 1, ti.max(0, j + dy))
                    spatial_weight = ti.exp(
                        -(dx * dx + dy * dy)
                        / (
                            2.0
                            * self.sigma_d[None]
                            * self.sigma_d[None]
                            * self.radius[None]
                        )
                        - (self.buffers[x, y] - color).norm_sqr()
                        / (
                            2.0
                            * self.sigma_r[None]
                            * self.sigma_r[None]
                            * self.radius[None]
                        )
                    )
                    filter_sum += spatial_weight * self.buffers[x, y]
                    weight_sum += spatial_weight

            if weight_sum > 0:
                self.temp_buffers[i, j] = filter_sum / weight_sum

            else:
                self.temp_buffers[i, j] = vec3(0.0)

        for i, j in self.buffers:
            self.buffers[i, j] = self.temp_buffers[i, j] * self.weight[
                None
            ] + self.buffers[i, j] * (1 - self.weight[None])


@ti.data_oriented
class JointBilateralFilter(BilateralFilter):
    def __init__(
        self,
        enabled: bool = False,
        radius: int = 2,
        weight: float = 0.1,
        sigma_d: float = 1.0,
        sigma_r: float = 1.0,
        sigma_z: float = 1.0,
        sigma_p: float = 1.0,
        sigma_n: float = 1.0,
        sigmal_a: float = 1.0,
    ) -> None:
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.sigma_z = ti.field(dtype=ti.f32, shape=())
        self.sigma_p = ti.field(dtype=ti.f32, shape=())
        self.sigma_n = ti.field(dtype=ti.f32, shape=())
        self.sigma_a = ti.field(dtype=ti.f32, shape=())

        self.sigma_z[None] = sigma_z
        self.sigma_p[None] = sigma_p
        self.sigma_n[None] = sigma_n
        self.sigma_a[None] = sigmal_a

        self.params: Dict = {}
        super().__init__(
            enabled=enabled,
            radius=radius,
            weight=weight,
            sigma_d=sigma_d,
            sigma_r=sigma_r,
        )

    def _name(self) -> str:
        return "Joint Bilateral Filter"

    def set_buffers(self, buffers) -> None:
        self.buffers = buffers

        self.res = self.buffers.shape
        self.temp_buffers = ti.Vector.field(3, dtype=ti.f32, shape=self.res)

    def set_sigma_z(self, sigma_z: float) -> None:
        self.sigma_z[None] = sigma_z
        self.update()

    def set_sigma_p(self, sigma_p: float) -> None:
        self.sigma_p[None] = sigma_p
        self.update()

    def set_sigma_n(self, sigma_n: float) -> None:
        self.sigma_n[None] = sigma_n
        self.update()

    def set_sigma_a(self, sigma_a: float) -> None:
        self.sigma_a[None] = sigma_a
        self.update()

    def update(self) -> None:
        super().update()
        self.params["radius"] = self.radius[None]
        self.params["weight"] = self.weight[None]
        self.params["sigma_z"] = self.sigma_z[None]
        self.params["sigma_p"] = self.sigma_p[None]
        self.params["sigma_n"] = self.sigma_n[None]
        self.params["sigma_a"] = self.sigma_a[None]

    def fetch_gbuffer(self, g_buffer) -> None:
        self.g_buffer = g_buffer

    @ti.kernel
    def process(self):
        for i, j in self.buffers:
            filter_sum = vec3(0.0)
            weight_sum = 0.0
            for dx in range(-self.radius[None], self.radius[None] + 1):
                x = ti.min(self.res[0] - 1, ti.max(0, i + dx))
                for dy in range(-self.radius[None], self.radius[None] + 1):
                    y = ti.min(self.res[1] - 1, ti.max(0, j + dy))
                    spatial_weight = ti.exp(
                        -(dx * dx + dy * dy)
                        / (
                            2.0
                            * self.sigma_d[None]
                            * self.sigma_d[None]
                            * self.radius[None]
                        )
                        - (self.buffers[x, y] - self.buffers[i, j]).norm_sqr()
                        / (
                            2.0
                            * self.sigma_r[None]
                            * self.sigma_r[None]
                            * self.radius[None]
                        )
                        - ti.abs(self.g_buffer.depth[x, y] - self.g_buffer.depth[i, j])
                        / (
                            2.0
                            * self.sigma_z[None]
                            * self.sigma_z[None]
                            * self.radius[None]
                        )
                        - (
                            self.g_buffer.normal[x, y] - self.g_buffer.normal[i, j]
                        ).norm_sqr()
                        / (
                            2.0
                            * self.sigma_a[None]
                            * self.sigma_a[None]
                            * self.radius[None]
                        )
                        - (self.g_buffer.pos[x, y] - self.g_buffer.pos[i, j]).norm_sqr()
                        / (
                            2.0
                            * self.sigma_p[None]
                            * self.sigma_p[None]
                            * self.radius[None]
                        )
                        - (
                            self.g_buffer.albedo[x, y] - self.g_buffer.albedo[i, j]
                        ).norm_sqr()
                        / (
                            2.0
                            * self.sigma_n[None]
                            * self.sigma_n[None]
                            * self.radius[None]
                        )
                    )
                    filter_sum += spatial_weight * self.buffers[x, y]
                    weight_sum += spatial_weight
            if weight_sum > 0:
                self.temp_buffers[i, j] = filter_sum / weight_sum
            else:
                self.temp_buffers[i, j] = vec3(0.0)
        for i, j in self.buffers:
            self.buffers[i, j] = self.temp_buffers[i, j] * self.weight[
                None
            ] + self.buffers[i, j] * (1 - self.weight[None])
