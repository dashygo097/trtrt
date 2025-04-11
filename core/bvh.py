from random import randint

import taichi as ti

from .records import BVHHitInfo
from .utils import EPSILON, TMAX, TMIN


@ti.dataclass
class Interval:
    min: ti.f32
    max: ti.f32

    @ti.func
    def from_fp(self, v0: ti.f32, v1: ti.f32):
        self.min = ti.min(v0, v1)
        self.max = ti.max(v0, v1)

    @ti.func
    def do_intersect(self, that):
        return Interval(min=ti.max(self.min, that.min), max=ti.min(self.max, that.max))

    @ti.func
    def union(self, that):
        return Interval(min=ti.min(self.min, that.min), max=ti.max(self.max, that.max))

    @ti.func
    def clamp(self, x: ti.f32):
        return ti.max(self.min, ti.min(x, self.max))

    @ti.func
    def pad(self, padding: ti.f32):
        self.min -= padding
        self.max += padding


@ti.kernel
def interval_union(this: Interval, that: Interval) -> Interval:
    return this.union(that)


@ti.dataclass
class AABB:
    x: Interval
    y: Interval
    z: Interval

    @ti.func
    def from_vec(self, v1, v2):
        self.x.from_fp(v1.x, v2.x)
        self.y.from_fp(v1.y, v2.y)
        self.z.from_fp(v1.z, v2.z)

    @ti.func
    def do_intersect(self, that):
        return AABB(
            x=self.x.do_intersect(that.x),
            y=self.y.do_intersect(that.y),
            z=self.z.do_intersect(that.z),
        )

    @ti.func
    def union(self, that):
        return AABB(
            x=self.x.union(that.x),
            y=self.y.union(that.y),
            z=self.z.union(that.z),
        )

    @ti.func
    def axis(self, n):
        ret = Interval()
        if n == 0:
            ret = self.x
        elif n == 1:
            ret = self.y
        elif n == 2:
            ret = self.z

        return ret

    @ti.func
    def intersect(self, ray, tmin=TMIN, tmax=TMAX):
        hit_tmin = tmin
        hit_tmax = tmax
        for i in range(3):
            ray_inv = 1.0 / ray.dir[i] + EPSILON
            t0 = (self.axis(i).min - ray.origin[i]) * ray_inv
            t1 = (self.axis(i).max - ray.origin[i]) * ray_inv

            if ray_inv < 0:
                t0, t1 = t1, t0

            hit_tmin = ti.max(t0, hit_tmin)
            hit_tmax = ti.min(t1, hit_tmax)

            if hit_tmax <= hit_tmin:
                hit_tmin = tmin
                hit_tmax = tmax

        return BVHHitInfo(tmin=hit_tmin, tmax=hit_tmax, is_hit=hit_tmax >= hit_tmin)


@ti.kernel
def aabb_union(this: AABB, that: AABB) -> AABB:
    return this.union(that)


@ti.kernel
def aabb_axis(this: AABB, n: ti.i32) -> Interval:
    return this.axis(n)


@ti.kernel
def get_centroid(obj: ti.template(), axis: ti.i32) -> ti.f32:
    interval = obj.bbox.axis(axis)
    return (interval.min + interval.max) / 2.0


@ti.data_oriented
class BVHNode:
    def __init__(self):
        self.bbox = None
        self.obj_id = -1

        self.left = None
        self.right = None

        self.leaf = ti.field(ti.i32, shape=())

        self.leaf[None] = 0
        self.has_left = 0
        self.has_right = 0

    def set_left(self, left):
        self.left = left
        self.has_left = 1

    def set_right(self, right):
        self.right = right
        self.has_right = 1

    def set_leaf(self, obj, obj_id):
        self.bbox = obj.bbox
        self.obj_id = obj_id
        self.leaf[None] = 1

    @ti.func
    def intersect(self, ray, tmin=TMIN, tmax=TMAX):
        is_hit = False
        hit_obj_id = -1

        bvh_hitinfo = self.bbox.intersect(ray, tmin, tmax)

        if bvh_hitinfo.is_hit:
            # If this is a leaf node, return the object ID
            if self.leaf[None] == 1:
                is_hit = True
                hit_obj_id = self.obj_id
            else:
                hit_dist = tmax
                if ti.static(self.has_left):
                    left_hit, left_obj_id = self.left.intersect(
                        ray, bvh_hitinfo.tmin, hit_dist
                    )
                    if left_hit:
                        is_hit = True
                        hit_obj_id = left_obj_id

                if ti.static(self.has_right):
                    right_hit, right_obj_id = self.right.intersect(
                        ray, bvh_hitinfo.tmin, hit_dist
                    )
                    if right_hit:
                        is_hit = True
                        hit_obj_id = right_obj_id

        return is_hit, hit_obj_id


def build_bvh(objects, start, end):
    if start >= end:
        return None

    if end - start == 1:
        node = BVHNode()
        node.set_leaf(objects[start], start)
        return node

    axis = randint(0, 2)

    objects[start:end] = sorted(
        objects[start:end],
        key=lambda obj: get_centroid(obj, axis),
    )

    bbox = objects[start].bbox
    for i in range(start + 1, end):
        bbox = aabb_union(bbox, objects[i].bbox)

    mid = (start + end) // 2
    node = BVHNode()
    left = build_bvh(objects, start, mid)
    right = build_bvh(objects, mid, end)

    if left is not None:
        node.set_left(left)

    if right is not None:
        node.set_right(right)

    node.bbox = bbox

    return node


@ti.func
def bvh_intersect(bvh, ray, tmin=TMIN, tmax=TMAX):
    is_hit, obj_id = bvh.intersect(ray, tmin, tmax)
    return is_hit, obj_id


def init_bbox(objects):
    from .objects import Sphere, Triangle, init_sphere, init_triangle

    num_objects = len(objects)
    for index in range(num_objects):
        if isinstance(objects[index], Triangle):
            init_triangle(objects[index])

        elif isinstance(objects[index], Sphere):
            init_sphere(objects[index])

        else:
            raise ValueError("Invalid object type, expected Triangle or Sphere")
