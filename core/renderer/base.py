from typing import Dict

import taichi as ti
from taichi.math import vec3

from ..objects import Ray
from ..records import GBuffer
from ..utils.const import FAR_Z, NEAR_Z, TMIN
from .sampler import Sampler


@ti.data_oriented
class Renderer:
    def __init__(
        self,
        sampler: Sampler = Sampler(),
        samples_per_pixel: int = 4,
    ) -> None:
        self.sampler = sampler
        self.samples_per_pixel = ti.field(dtype=ti.i32, shape=())

        self.samples_per_pixel[None] = samples_per_pixel

        self.params: Dict = {}
        self.update()

    def _name(self):
        return self.__class__.__name__

    def set_sampler(self, sampler: Sampler) -> None:
        self.sampler = sampler
        self.update()

    def set_spp(self, spp: int) -> None:
        self.samples_per_pixel[None] = spp
        self.update()

    def get_params(self) -> Dict:
        return self.params

    def update(self) -> None:
        # Update params
        self.params["sampler"] = self.sampler._name()
        self.params["samples_per_pixel"] = self.samples_per_pixel[None]

    @ti.func
    def ray_color(self, scene, ray: Ray):  # pyright: ignore
        color_buffer = ti.Vector([1.0, 1.0, 1.0])
        return color_buffer

    @ti.func
    def fetch_gbuffer(self, scene, ray: Ray):  # pyright: ignore
        hitinfo = scene.intersect(ray)
        # TODO: A better way wo impl depth buffer
        depth = (hitinfo.time - NEAR_Z) / (FAR_Z - NEAR_Z)
        return GBuffer(
            depth=depth,
            pos=hitinfo.pos,
            normal=ti.abs(1 + hitinfo.normal) * 0.5,
            albedo=hitinfo.albedo,
        )

    @ti.func
    def sample_direct_light(self, scene, hit_point, hit_normal):
        direct_light = vec3(0.0)
        light_pos = vec3(0.0)
        light_dir = vec3(0.0)

        if scene.light_ptr[None] > 0:
            index = ti.random(ti.i32) % scene.light_ptr[None]
            index = scene.light_map[index]

            light_normal = vec3(0.0)
            light_color = vec3(0.0)

            if index < scene.tri_ptr[None]:
                light = scene.mesh[index]
                light_pos = light.sample_point()
                light_normal = light.get_normal(light_pos)
                light_color = light.emission
            else:
                light = scene.spheres[index - scene.tri_ptr[None]]
                light_pos = light.sample_point()
                light_normal = light.get_normal(light_pos)
                light_color = light.emission

            dir_noise = self.sampler.hemispherical_sample(light_normal)
            light_dir = (light_pos - hit_point + dir_noise).normalized()
            distance = (light_pos - hit_point).norm()

            shadow_ray = Ray(hit_point + light_dir * TMIN, light_dir)
            hitinfo = scene.intersect(shadow_ray, TMIN, distance - 2 * TMIN)

            if not hitinfo.is_hit:
                cos_theta_surf = max(0.0, light_dir.dot(hit_normal))
                cos_theta_light = max(0.0, -light_dir.dot(light_normal))

                geometry_term = (
                    cos_theta_surf * cos_theta_light / max(TMIN, distance * distance)
                )

                light_pdf = 1.0 / scene.light_ptr[None]

                direct_light = light_color * geometry_term / light_pdf

        return Ray(origin=hit_point, dir=light_dir), direct_light

    @ti.func
    def sample_directional_light(self, scene, pos, normal):
        dir_light = scene.dir_light.color
        light_dir = -scene.dir_light.dir
        # Add soft shadow
        light_dir = self.sampler.sample_cone(light_dir, 5.0)

        cos_theta = ti.max(0.0, normal.dot(light_dir))

        if cos_theta <= 0.0:
            dir_light = vec3(0.0)

        else:
            shadow_ray = Ray(pos, light_dir)
            shadow_info = scene.intersect(shadow_ray)

            if (
                shadow_info.is_hit
                and shadow_info.time > TMIN
                and shadow_info.emission.norm() == 0.0
            ):
                dir_light = vec3(0.0)

        return dir_light * cos_theta

    @ti.func
    def sample_light_ray(self, scene):
        light_pos = ti.Vector([0.0, 0.0, 0.0])
        light_normal = ti.Vector([0.0, 0.0, 0.0])
        light_dir = ti.Vector([0.0, 0.0, 0.0])

        ray = Ray()

        if scene.light_ptr[None] == 0:
            pass

        else:
            # NOTE: SEEMING BAD CODE BUT IT WORKS (OTHERWISE IT RAISES ERROR)
            index = ti.random(ti.i32) % scene.light_ptr[None]
            index = scene.light_map[index]
            if index < scene.tri_ptr[None]:
                light = scene.mesh[index]
                light_pos = light.sample_point()
                light_normal = light.get_normal(light_pos)
            else:
                light = scene.spheres[index - scene.tri_ptr[None]]
                light_pos = light.sample_point()
                light_normal = light.get_normal(light_pos)

            light_dir = self.sampler.hemispherical_sample(light_normal)
            ray = Ray(light_pos, light_dir)

        return ray

    @ti.func
    def is_visible(self, scene, p1, p2) -> bool:
        direction = (p2 - p1).normalized()
        distance = (p2 - p1).norm()
        shadow_ray = Ray(p1 + direction * TMIN, direction)

        hitinfo = scene.intersect(shadow_ray, TMIN, distance)

        return not hitinfo.is_hit
