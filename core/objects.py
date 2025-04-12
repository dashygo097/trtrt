import taichi as ti
from taichi.math import vec3

from .records import HitInfo
from .utils.const import EPSILON, FAR_Z, NEAR_Z, TMAX, TMIN


@ti.dataclass
class Ray:
    origin: vec3  # pyright: ignore
    dir: vec3  # pyright: ignore

    @ti.func
    def at(self, t):
        return self.origin + t * self.dir


@ti.dataclass
class Triangle:
    tag: ti.i32  # pyright: ignore
    v0: vec3  # pyright: ignore
    v1: vec3  # pyright: ignore
    v2: vec3  # pyright: ignore

    albedo: vec3  # pyright: ignore
    metallic: ti.f32  # pyright: ignore
    roughness: ti.f32  # pyright: ignore
    emission: vec3  # pyright: ignore

    @ti.func
    def intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX):  # pyright: ignore
        """
        Moller-Trumbore algorithm for triangle-ray intersection
        """
        is_hit = False
        time = TMAX
        hit_pos = vec3(0.0)
        hit_normal = vec3(0.0)
        hit_front = False

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
            albedo=self.albedo,
            metallic=self.metallic,
            roughness=self.roughness,
            emission=self.emission,
        )

    @ti.func
    def sample_point(self):
        u = ti.random()
        v = ti.random()
        w = ti.sqrt(u)

        u = 1 - w
        v *= w
        return u * self.v0 + v * self.v1 + (1 - u - v) * self.v2

    @ti.func
    def get_normal(self, pos):
        sign = 1 if ti.random() > 0.5 else -1
        return (self.v1 - self.v0).cross(self.v2 - self.v0).normalized() * sign


@ti.dataclass
class Sphere:
    tag: ti.i32  # pyright: ignore
    center: vec3  # pyright: ignore
    radius: float  # pyright: ignore

    albedo: vec3  # pyright: ignore
    metallic: ti.f32  # pyright: ignore
    roughness: ti.f32  # pyright: ignore
    emission: vec3  # pyright: ignore

    @ti.func
    def intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX):  # pyright: ignore
        """
        Ray-sphere intersection
        """
        is_hit = False
        time = TMAX
        hit_pos = vec3(0.0)
        hit_normal = vec3(0.0)
        hit_front = False

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
                if hit_front:
                    hit_normal = -hit_normal
        return HitInfo(
            is_hit=is_hit,
            time=time,
            pos=hit_pos,
            normal=hit_normal,
            front=hit_front,
            tag=self.tag,
            albedo=self.albedo,
            metallic=self.metallic,
            roughness=self.roughness,
            emission=self.emission,
        )

    @ti.func
    def sample_point(self):
        theta = ti.random() * 2 * ti.math.pi
        phi = ti.random() * ti.math.pi
        x = self.center.x + self.radius * ti.sin(phi) * ti.cos(theta)
        y = self.center.y + self.radius * ti.sin(phi) * ti.sin(theta)
        z = self.center.z + self.radius * ti.cos(phi)
        return vec3(x, y, z)

    @ti.func
    def get_normal(self, pos):
        return (pos - self.center).normalized()


@ti.dataclass
class DirecLight:
    dir: vec3  # pyright: ignore
    color: vec3  # pyright: ignore
