import taichi as ti
from taichi.math import vec3

from ..geometry.bvh import AABB
from ..records import HitInfo
from ..utils.const import EPSILON, TMAX, TMIN
from .lights import Ray
from .pbr import PBRMaterial


@ti.dataclass
class Triangle:
    tag: ti.i32
    v0: vec3
    v1: vec3
    v2: vec3
    bbox: AABB

    pbr: PBRMaterial

    @ti.func
    def intersect(self, ray: Ray, tmin: ti.f32 = TMIN, tmax: ti.f32 = TMAX) -> HitInfo:
        """
        Moller-Trumbore algorithm for triangle-ray intersection
        """
        is_hit = False
        time = TMAX
        hit_pos = vec3(0.0)
        hit_normal = vec3(0.0)
        hit_front = False
        u = 0.0
        v = 0.0

        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        oc = ray.origin - self.v0
        s1 = ray.dir.cross(edge2)
        s2 = oc.cross(edge1)
        divisor = s1.dot(edge1)
        if divisor == 0:
            pass
        else:
            t = s2.dot(edge2) / divisor
            b1 = s1.dot(oc) / divisor
            b2 = s2.dot(ray.dir) / divisor
            if b1 >= 0 and b2 >= 0 and b1 + b2 <= 1 and t > tmin and t < tmax:
                is_hit = True
                time = t
                hit_pos = ray.at(t)
                hit_normal = edge1.cross(edge2).normalized()
                hit_front = ray.dir.dot(hit_normal) > 0
                if hit_front:
                    hit_normal = -hit_normal
        return HitInfo(
            is_hit=is_hit,
            time=time,
            pos=hit_pos,
            normal=hit_normal,
            front=hit_front,
            tag=self.tag,
            u=u,
            v=v,
            albedo=self.pbr.albedo,
            metallic=self.pbr.metallic,
            roughness=self.pbr.roughness,
            emission=self.pbr.emission,
        )

    @ti.func
    def sample_point(self) -> vec3:
        u = ti.random()
        v = ti.random()
        w = ti.sqrt(u)

        u = 1 - w
        v *= w
        return u * self.v0 + v * self.v1 + (1 - u - v) * self.v2

    @ti.func
    def sample_certain_point(self, u: ti.f32, v: ti.f32) -> vec3:
        w = ti.sqrt(u)
        u = 1 - w
        v *= w
        return u * self.v0 + v * self.v1 + (1 - u - v) * self.v2

    @ti.func
    def normal(self, pos: vec3) -> vec3:
        sign = 1 if ti.random() > 0.5 else -1
        return (self.v1 - self.v0).cross(self.v2 - self.v0).normalized() * sign

    @ti.func
    def centroid(self) -> vec3:
        return (self.v0 + self.v1 + self.v2) / 3.0


@ti.dataclass
class Sphere:
    tag: ti.i32
    center: vec3
    radius: ti.f32
    bbox: AABB

    pbr: PBRMaterial

    @ti.func
    def intersect(self, ray: Ray, tmin: ti.f32 = TMIN, tmax: ti.f32 = TMAX) -> HitInfo:
        """
        Ray-sphere intersection
        """
        is_hit = False
        time = TMAX
        hit_pos = vec3(0.0)
        hit_normal = vec3(0.0)
        hit_front = False
        u = 0.0
        v = 0.0

        oc = ray.origin - self.center
        a = ray.dir.dot(ray.dir)
        b = 2.0 * oc.dot(ray.dir)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - 4 * a * c

        if discriminant > 0:
            # FIXIT: Warning: a might be zero
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a + EPSILON)
            if t > tmin and t < tmax:
                is_hit = True
                time = t
                hit_pos = ray.at(t)
                hit_normal = (hit_pos - self.center).normalized()
                hit_front = ray.dir.dot(hit_normal) > 0
                u = 0.5 + ti.atan2(hit_normal.z, hit_normal.x) / (2 * ti.math.pi)
                v = 0.5 - ti.asin(hit_normal.y) / ti.math.pi
                if hit_front:
                    hit_normal = -hit_normal
        return HitInfo(
            is_hit=is_hit,
            time=time,
            pos=hit_pos,
            normal=hit_normal,
            front=hit_front,
            tag=self.tag,
            u=u,
            v=v,
            albedo=self.pbr.albedo,
            metallic=self.pbr.metallic,
            roughness=self.pbr.roughness,
            emission=self.pbr.emission,
        )

    @ti.func
    def sample_point(self) -> vec3:
        theta = ti.random() * 2 * ti.math.pi
        phi = ti.random() * ti.math.pi
        x = self.center.x + self.radius * ti.sin(phi) * ti.cos(theta)
        y = self.center.y + self.radius * ti.sin(phi) * ti.sin(theta)
        z = self.center.z + self.radius * ti.cos(phi)
        return vec3(x, y, z)

    @ti.func
    def sample_certain_point(self, u: ti.f32, v: ti.f32) -> vec3:
        theta = u * 2 * ti.math.pi
        phi = v * ti.math.pi
        x = self.center.x + self.radius * ti.sin(phi) * ti.cos(theta)
        y = self.center.y + self.radius * ti.sin(phi) * ti.sin(theta)
        z = self.center.z + self.radius * ti.cos(phi)
        return vec3(x, y, z)

    @ti.func
    def normal(self, pos: vec3) -> vec3:
        return (pos - self.center).normalized()

    @ti.func
    def centroid(self) -> vec3:
        return self.center
