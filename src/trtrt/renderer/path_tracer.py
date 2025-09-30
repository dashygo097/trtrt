from typing import Dict

import taichi as ti
from taichi.math import vec3

from ..objects import Ray
from ..utils import EPSILON, TMIN, ObjectTag
from .base import Renderer
from .sampler import Sampler, UniformSampler
from .utils import (direct_remapping, geometry_smith, ggx_distribution,
                    reflect, refract, schlick_fresnel)


@ti.data_oriented
class PathTracer(Renderer):
    def __init__(
        self,
        sampler: Sampler = UniformSampler(),
        samples_per_pixel: int = 4,
        max_depth: int = 1,
        ambient_rate: float = 0.1,
        direct_light_weight: float = 1.0,
        p_rr: float = 0.8,
    ) -> None:
        self.max_depth = ti.field(dtype=ti.i32, shape=())
        self.ambient_rate = ti.field(dtype=ti.f32, shape=())
        self.direct_light_weight = ti.field(dtype=ti.f32, shape=())
        self.p_rr = ti.field(dtype=ti.f32, shape=())

        self.max_depth[None] = max_depth
        self.ambient_rate[None] = ambient_rate
        self.direct_light_weight[None] = direct_light_weight
        self.p_rr[None] = p_rr

        self.params: Dict = {}
        super().__init__(sampler, samples_per_pixel)
        self.update()

    def _name(self) -> str:
        return "Path Tracer"

    def set_max_depth(self, max_depth: int) -> None:
        self.max_depth[None] = max_depth
        self.update()

    def set_direct_light_weight(self, direct_light_weight: float) -> None:
        self.direct_light_weight[None] = direct_light_weight
        self.update()

    def set_prr(self, p_rr: float) -> None:
        self.p_rr[None] = p_rr
        self.update()

    def set_ambient_rate(self, ambient_rate: float) -> None:
        self.ambient_rate[None] = ambient_rate
        self.update()

    def update(self) -> None:
        # Update params
        super().update()
        self.params["max_depth"] = self.max_depth[None]
        self.params["ambient_rate"] = self.ambient_rate[None]
        self.params["direct_light_weight"] = self.direct_light_weight[None]
        self.params["p_rr"] = self.p_rr[None]

    @ti.func
    def _get_light_dir_noise(self, dir: vec3, noise: ti.f32):
        return (
            dir
            + ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5])
            * noise
        ).normalized()

    @ti.func
    def ray_color(self, scene, ray: Ray, _u: ti.f32, _v: ti.f32) -> vec3:
        color_buffer = vec3(0.0)
        luminance = vec3(1.0)

        for bounce in range(self.max_depth[None]):
            # Russian Roulette
            if bounce > 0 and ti.random() > self.p_rr[None]:
                break

            if bounce > 0:
                luminance /= self.p_rr[None]

            hitinfo = scene.intersect(ray)

            if hitinfo.is_hit and hitinfo.time > TMIN:
                # PBR: Direct lighting
                if hitinfo.tag == ObjectTag.PBR:
                    N = hitinfo.normal
                    V = -ray.dir
                    F0 = (
                        vec3(0.4) * (1.0 - hitinfo.metallic)
                        + hitinfo.albedo * hitinfo.metallic
                    )
                    NdotV = max(N.dot(V), 0.0)
                    alpha = hitinfo.roughness * hitinfo.roughness

                    if self.direct_light_weight[None] > 0.0:
                        # NOTE: That the light_color is zero is equivalent to not is_visible()
                        light_ray, light_color = self.sample_direct_light(
                            scene, hitinfo.pos, hitinfo.normal, _u, _v
                        )
                        L = light_ray.dir
                        NdotL = ti.max(hitinfo.normal.dot(L), 0.0)

                        if NdotL > 0.0:
                            H = (V + L).normalized()
                            VdotL = ti.max(V.dot(L), 0.0)
                            HdotV = ti.max(H.dot(V), 0.0)

                            k = direct_remapping(alpha)
                            NDF = ggx_distribution(hitinfo.normal, H, hitinfo.roughness)
                            G = geometry_smith(NdotV, VdotL, k)
                            F = schlick_fresnel(HdotV, F0)

                            ks = F
                            kd = 1.0 - ks
                            kd *= 1.0 - hitinfo.metallic

                            nominator = NDF * G * F
                            denom = (4 * NdotV * NdotL) + EPSILON
                            specular = nominator / denom

                            color_buffer += (
                                (kd * hitinfo.albedo / ti.math.pi + specular)
                                * luminance
                                * NdotL
                                * self.direct_light_weight[None]
                            ) * light_color

                    # PBR: Reflection
                    is_specular = False
                    scatter_dir = vec3(0.0)
                    perfect_reflect = reflect(ray.dir, hitinfo.normal)

                    if ti.random() < hitinfo.metallic:
                        # Specular reflection
                        is_specular = True
                        if hitinfo.roughness > 0.0:
                            scatter_dir = self.sampler.ggx_sample(
                                V, hitinfo.normal, alpha, _u, _v
                            )
                        else:
                            scatter_dir = perfect_reflect
                    else:
                        # Diffuse reflection
                        is_specular = False
                        scatter_dir = self.sampler.hemispherical_sample(
                            hitinfo.normal, _u, _v
                        )

                    if is_specular:
                        luminance *= hitinfo.albedo

                    else:
                        cos_term = max(0.0, scatter_dir.dot(hitinfo.normal))
                        luminance *= hitinfo.albedo * cos_term * ti.math.pi
                    ray = Ray(hitinfo.pos, scatter_dir)  # Has normlalized

                    # PBR: Emission
                    color_buffer += luminance * hitinfo.emission

                elif hitinfo.tag == ObjectTag.GLASS:
                    # Glass
                    refract_dir = refract(ray.dir, hitinfo.normal, hitinfo.refraction)
                    ray = Ray(hitinfo.pos, refract_dir)
                    luminance *= hitinfo.albedo

            else:
                # Background
                background_color = scene.ambient_light.color
                if bounce > 0:
                    color_buffer += (
                        background_color * luminance * self.ambient_rate[None]
                    )

                    # Directional light
                    directional_light = self.sample_directional_light(
                        scene, ray.origin, ray.dir, _u, _v
                    )
                    color_buffer += (
                        directional_light * luminance * self.direct_light_weight[None]
                    )
                    break

                else:
                    color_buffer += background_color * luminance
                    break

        return color_buffer
