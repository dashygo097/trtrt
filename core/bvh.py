from operator import is_
from random import randint
from typing import List

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
        is_hit = False
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

        is_hit = hit_tmax >= hit_tmin

        return BVHHitInfo(is_hit=is_hit, tmin=hit_tmin, tmax=hit_tmax)


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


def sort_objects(objects: List) -> None:
    axis = randint(0, 2)
    objects.sort(key=lambda obj: get_centroid(obj, axis))


@ti.dataclass
class BVHNode:
    bbox: AABB
    left_idx: ti.i32  # Index of left child in node array
    right_idx: ti.i32  # Index of right child in node array
    object_idx: ti.i32  # Index of object (if leaf node)
    is_leaf: ti.i32  # Using i32 as bool (0=False, 1=True)


@ti.data_oriented
class Stack:
    def __init__(self, max_size):
        self.max_size = max_size
        self.stack = ti.field(ti.i32, shape=max_size)
        self.size = ti.field(ti.i32, shape=())

    @ti.func
    def push(self, value: ti.i32):
        if self.size[None] < self.max_size:
            self.stack[self.size[None]] = value
            self.size[None] += 1

    @ti.func
    def pop(self) -> ti.i32:
        obj = -1
        if self.size[None] > 0:
            self.size[None] -= 1
            obj = self.stack[self.size[None]]
        return obj


@ti.data_oriented
class BVH:
    def __init__(self, objects):
        self.objects = objects

        self.max_nodes = 2 * len(objects)

        self.nodes = BVHNode.field(shape=self.max_nodes)
        self.nodes_used = ti.field(ti.i32, shape=())
        self.nodes_used[None] = 0

        self.root_idx = self.build(objects, 0, len(objects))

        self.stack = Stack(self.max_nodes)

    def build(self, objects, start, end):
        if start >= end:
            return -1

        node_idx = self.nodes_used[None]
        self.nodes_used[None] += 1

        node = BVHNode(bbox=AABB(), left_idx=-1, right_idx=-1, object_idx=-1, is_leaf=0)

        if end - start == 1:
            node.bbox = objects[start].bbox
            node.object_idx = start
            node.is_leaf = 1
            self.nodes[node_idx] = node
            return node_idx

        node.bbox = objects[start].bbox
        for i in range(start + 1, end):
            node.bbox = aabb_union(node.bbox, objects[i].bbox)

        objects_slice = objects[start:end]
        sort_objects(objects_slice)
        for i in range(len(objects_slice)):
            objects[start + i] = objects_slice[i]

        mid = start + (end - start) // 2

        node.left_idx = self.build(objects, start, mid)
        node.right_idx = self.build(objects, mid, end)

        self.nodes[node_idx] = node
        return node_idx

    def info(self):
        pass

    @ti.func
    def intersect(self, scene, ray: ti.template()) -> BVHHitInfo:
        bvh_hitinfo = BVHHitInfo(is_hit=False, tmin=TMIN, tmax=TMAX, obj_id=-1)
        self.stack.push(self.root_idx)

        while self.stack.size[None] > 0:
            node_idx = self.stack.pop()

            node = self.nodes[node_idx]
            box_hit = node.bbox.intersect(ray, bvh_hitinfo.tmin, bvh_hitinfo.tmax)

            if box_hit.is_hit:
                if node.is_leaf:
                    obj_hit = node.bbox.intersect(
                        ray, bvh_hitinfo.tmin, bvh_hitinfo.tmax
                    )
                    if obj_hit.is_hit and obj_hit.tmin < bvh_hitinfo.tmax:
                        bvh_hitinfo.is_hit = True
                        bvh_hitinfo.tmin = obj_hit.tmin
                        bvh_hitinfo.tmax = obj_hit.tmax
                        bvh_hitinfo.obj_id = node.object_idx
                        break
                else:
                    self.stack.push(node.right_idx)
                    self.stack.push(node.left_idx)

        return bvh_hitinfo


def init_bbox(objects):
    from .objects import init4bbox

    num_objects = len(objects)
    for index in range(num_objects):
        init4bbox(objects[index])
