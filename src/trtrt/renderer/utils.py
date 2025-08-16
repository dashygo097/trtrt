import taichi as ti
from taichi.math import vec3

from ..utils.const import EPSILON


@ti.func
def reflect(v: vec3, n: vec3) -> vec3:
    return v - 2 * v.dot(n) * n


@ti.func
def refract(v: vec3, n: vec3, ior: ti.f32) -> vec3:
    dir = vec3(0.0)
    cos_theta = -v.dot(n)
    eta = 1.0 / ior if cos_theta > 0 else ior
    normal = n if cos_theta > 0 else -n
    cos_theta = ti.abs(cos_theta)

    k = 1 - eta * eta * (1 - cos_theta * cos_theta)
    if k < 0:
        dir = reflect(v, n)
    else:
        dir = eta * v + (eta * cos_theta - ti.sqrt(k)) * normal

    return dir


@ti.func
def schlick_fresnel(cos_theta: ti.f32, ior: ti.f32) -> ti.f32:
    r0 = ((1 - ior) / (1 + ior)) ** 2
    return r0 + (1 - r0) * (1 - cos_theta) ** 5


@ti.func
def ggx_distribution(n: vec3, h: vec3, roughness: ti.f32) -> ti.f32:
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    NdotH = ti.max(n.dot(h), 0.0)
    NdotH2 = NdotH * NdotH

    denom = ti.max(NdotH2 * (alpha2 - 1) + 1, 0.0)
    denom = ti.math.pi * denom * denom

    return alpha2 / ti.max(denom, EPSILON)


@ti.func
def geometry_schlick_ggx(ndotv, k: ti.f32) -> ti.f32:
    return ndotv / (ndotv * (1 - k) + k)


@ti.func
def geometry_smith(ndotv, vdotl, k: ti.f32) -> ti.f32:
    ggx_v = geometry_schlick_ggx(ndotv, k)
    ggx_l = geometry_schlick_ggx(vdotl, k)
    return ggx_v * ggx_l


@ti.func
def direct_remapping(alpha: ti.f32) -> ti.f32:
    return (1 + alpha) * (1 + alpha) / 8.0


@ti.func
def ibr_remapping(alpha: ti.f32) -> ti.f32:
    return alpha * alpha / 2.0
