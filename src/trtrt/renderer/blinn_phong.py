from typing import Dict

import taichi as ti
from taichi.math import vec3

from ..objects import Ray
from ..utils import EPSILON, ObjectTag
from .base import Renderer
from .sampler import Sampler, UniformSampler


@ti.data_oriented
class BlinnPhong(Renderer):
    def __init__(
        self,
        sampler: Sampler = UniformSampler(),
        samples_per_pixel: int = 4,
        diffuse_rate: float = 1.0,
        ambient_rate: float = 0.0,
        enable_cosine: bool = True,
    ) -> None:
        self.diffuse_rate = ti.field(dtype=ti.f32, shape=())
        self.ambient_rate = ti.field(dtype=ti.f32, shape=())
        self.enable_cosine = ti.field(dtype=ti.i32, shape=())

        self.diffuse_rate[None] = diffuse_rate
        self.ambient_rate[None] = ambient_rate
        self.enable_cosine[None] = enable_cosine

        self.params: Dict = {}
        super().__init__(sampler, samples_per_pixel)
        self.update()

    def _name(self) -> str:
        return "BlinnPhong"

    def set_diffuse_rate(self, diffuse_rate: float) -> None:
        self.diffuse_rate[None] = diffuse_rate
        self.update()

    def set_ambient_rate(self, ambient_rate: float) -> None:
        self.ambient_rate[None] = ambient_rate
        self.update()

    def set_enable_cosine(self, enable_cosine: bool) -> None:
        self.enable_cosine[None] = enable_cosine
        self.update()

    def update(self) -> None:
        # Update params
        super().update()
        self.params["diffuse_rate"] = self.diffuse_rate[None]
        self.params["ambient_rate"] = self.ambient_rate[None]
        self.params["enable_cosine"] = self.enable_cosine[None]

    # NOTE: Considering performance, it is recommanded to write all the sampler into one big func/kernel
    @ti.func
    def sample_diffuse_light(
        self, scene, pos: vec3, normal: vec3, u: ti.f32, v: ti.f32
    ):
        diffuse = vec3(0.0)
        num_lights = scene.light_ptr
        if num_lights > 0:
            for i in range(num_lights):
                index = scene.light_map[i]
                light_pos = vec3(0.0)
                light_color = vec3(0.0)

                if index < scene.tri_ptr:
                    light = scene.mesh[index]
                    light_pos = light.sample_point()
                    light_color = light.pbr.emission
                else:
                    light = scene.spheres[index - scene.tri_ptr]
                    light_pos = light.sample_point()
                    light_color = light.pbr.emission

                light_dir = (light_pos - pos).normalized()
                distance = (light_pos - pos).norm()
                intensity = light_color / (distance * distance)
                if self.enable_cosine[None]:
                    cos_theta = ti.max(light_dir.dot(normal), 0.0)
                    intensity = intensity * cos_theta

                diffuse += intensity * self.diffuse_rate[None]

        return diffuse

    @ti.func
    def ray_color(self, scene, ray: Ray, _u: ti.f32, _v: ti.f32) -> vec3:
        color_buffer = vec3(0.0)
        hitinfo = scene.intersect(ray)
        if hitinfo.is_hit:
            if hitinfo.tag == ObjectTag.PBR:
                if hitinfo.emission.norm() <= EPSILON:
                    # ambient
                    t = 0.5 * (ray.dir.normalized()[1] + 1.0)
                    background_color = (1.0 - t) * scene.bg_bottom + t * scene.bg_top
                    color_buffer += background_color * self.ambient_rate[None]

                    diffuse_light = self.sample_diffuse_light(
                        scene, hitinfo.pos, hitinfo.normal, hitinfo.u, hitinfo.v
                    )
                    color_buffer += diffuse_light * hitinfo.albedo

                else:
                    color_buffer += hitinfo.albedo

        else:
            # Background
            t = 0.5 * (ray.dir.normalized()[1] + 1.0)
            background_color = (1.0 - t) * scene.bg_bottom + t * scene.bg_top
            color_buffer += background_color

        return color_buffer
