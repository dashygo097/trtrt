from random import randint
from typing import List, Optional

import taichi as ti
from taichi.math import vec3

from .records import BVHHitInfo
from .utils.const import EPSILON, TMAX, TMIN


@ti.dataclass
class AABB:
    min: vec3
    max: vec3

    def union(self, other):
        return AABB(
            min=ti.min(self.min, other.min),
            max=ti.max(self.max, other.max),
        )

    @ti.func
    def intersect(self, ray):
        inv_dir = 1.0 / (ray.dir + EPSILON)
        t0 = (self.min - ray.origin) * inv_dir
        t1 = (self.max - ray.origin) * inv_dir
        t_min = ti.max(ti.min(t0, t1))
        t_max = ti.min(ti.max(t0, t1))
        return t_min, t_max


@ti.func
def bbox_valid(tmin: vec3, tmax: vec3) -> bool:
    return (tmin.x <= tmax.x) and (tmin.y <= tmax.y) and (tmin.z <= tmax.z)


def get_centroid(aabb: AABB) -> vec3:
    return (aabb.min + aabb.max) * 0.5


def sort_objects(objects: List, axis: int) -> List:
    return sorted(objects, key=lambda obj: get_centroid(obj.bbox)[axis])


@ti.dataclass
class BVHNode:
    aabb: AABB
    left_id: ti.i32
    right_id: ti.i32
    obj_id: ti.i32


@ti.data_oriented
class BVH:
    def __init__(self, objects: Optional[List] = None) -> None:
        self.objects = objects.copy() if objects is not None else []
        self.set_objects(self.objects)

    def set_objects(self, objects: List) -> None:
        self.objects = objects.copy()
        max_nodes = len(self.objects) * 2 + 1
        self.nodes = BVHNode.field(shape=(max_nodes,))
        self.nodes_id = ti.field(ti.i32, shape=())
        self.nodes_id[None] = 0
        self.root_id = -1

    def build(self) -> None:
        self.root_id = self._build(self.objects, 0, len(self.objects))

    def _build(self, objects: List, start: int, end: int) -> int:
        nodes_id = self.nodes_id[None]
        self.nodes_id[None] += 1

        if end - start == 1:
            obj = objects[start]  # Use start index, not nodes_id
            self.nodes[nodes_id] = BVHNode(
                aabb=obj.bbox,
                left_id=-1,
                right_id=-1,
                obj_id=start,
            )
            return nodes_id

        axis = randint(0, 2)
        objects[start:end] = sort_objects(objects[start:end], axis)

        mid = start + (end - start) // 2

        self.nodes[nodes_id] = BVHNode(
            aabb=AABB(vec3(0), vec3(0)),
            left_id=-1,
            right_id=-1,
            obj_id=-1,
        )

        for i in range(start, end):
            self.nodes[nodes_id].aabb = self.nodes[nodes_id].aabb.union(objects[i].bbox)

        left_id = self._build(objects, start, mid)
        right_id = self._build(objects, mid, end)

        self.nodes[nodes_id].left_id = left_id
        self.nodes[nodes_id].right_id = right_id

        return nodes_id

    def info(self) -> None:
        print(f"BVH Info:")
        print(f"  Number of objects: {len(self.objects)}")
        print(f"  Number of nodes: {self.nodes_id[None]}")
        print(f"  Root node ID: {self.root_id}")

        depths = []
        leaf_counts = [0]
        internal_counts = [0]

        def traverse(node_id, depth=0):
            if node_id == -1:
                return

            node = self.nodes[node_id]

            if node.left_id == -1 and node.right_id == -1:
                leaf_counts[0] += 1
                depths.append(depth)
            else:
                internal_counts[0] += 1
                traverse(node.left_id, depth + 1)
                traverse(node.right_id, depth + 1)

        traverse(self.root_id)

        if depths:
            print(f"  Tree statistics:")
            print(f"    Leaf nodes: {leaf_counts[0]}")
            print(f"    Internal nodes: {internal_counts[0]}")
            print(f"    Maximum depth: {max(depths) if depths else 0}")
            print(
                f"    Average depth: {sum(depths) / len(depths) if depths else 0:.2f}"
            )

    def pretty_print(self) -> None:
        print("BVH Tree Structure:")

        def print_node(node_id, indent="", is_last=True):
            if node_id == -1:
                return

            node = self.nodes[node_id]

            branch = "└── " if is_last else "├── "
            print(f"{indent}{branch}Node {node_id}", end="")

            if node.obj_id != -1:
                print(f" (Leaf, Object: {node.obj_id})")
            else:
                print(f" (Internal)")

            new_indent = indent + ("    " if is_last else "│   ")

            if node.left_id != -1:
                has_right = node.right_id != -1
                print_node(node.left_id, new_indent, not has_right)
            if node.right_id != -1:
                print_node(node.right_id, new_indent, True)

        print_node(self.root_id)
