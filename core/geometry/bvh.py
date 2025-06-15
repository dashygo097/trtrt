from random import randint
from typing import List, Optional

import taichi as ti
from taichi.math import vec3

from ..records import BVHHitInfo
from ..utils.const import TMAX, TMIN, ObjectShape


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
    def intersect(self, ray, tmin=TMIN, tmax=TMAX):
        is_hit = True
        box_tmin = tmin
        box_tmax = tmax
        for i in range(3):
            inv_d = 1.0 / ray.dir[i]
            t0 = (self.min[i] - ray.origin[i]) * inv_d
            t1 = (self.max[i] - ray.origin[i]) * inv_d

            if t0 < t1:
                box_tmin = ti.max(t0, box_tmin)
                box_tmax = ti.min(t1, box_tmax)

            else:
                box_tmin = ti.max(t1, box_tmin)
                box_tmax = ti.min(t0, box_tmax)

            is_hit = is_hit and (box_tmin <= box_tmax)

        return BVHHitInfo(
            is_hit=is_hit,
            tmin=box_tmin,
            tmax=box_tmax,
        )


@ti.dataclass
class BVHNode:
    aabb: AABB
    left_id: ti.i32
    right_id: ti.i32
    obj_id: ti.i32


@ti.data_oriented
class BVH:
    def __init__(self, objects: Optional[List] = None) -> None:
        self.objects: List = objects if objects is not None else []
        self.labels: List = []
        self.set_objects(self.objects)

    def set_objects(self, objects: List) -> None:
        self.objects = objects
        self.labels = []
        max_nodes = len(self.objects) * 2 + 1
        self.nodes = BVHNode.field(shape=(max_nodes,))

        self.used_nodes = 0
        self.root_id = -1

    def build(self) -> None:
        self.root_id = self._build(self.objects, 0, len(self.objects))

    def _build(self, objects: List, start: int, end: int) -> int:
        used_nodes = self.used_nodes
        self.used_nodes += 1

        if end - start == 1:
            obj = objects[start].entity  # Use start index, not used_nodes
            self.nodes[used_nodes] = BVHNode(
                aabb=obj.bbox,
                left_id=-1,
                right_id=-1,
                obj_id=start,
            )
            return used_nodes

        axis = randint(0, 2)
        objects[start:end] = sort_objects(objects[start:end], axis)

        mid = start + (end - start) // 2

        self.nodes[used_nodes] = BVHNode(
            aabb=AABB(vec3(0), vec3(0)),
            left_id=-1,
            right_id=-1,
            obj_id=-1,
        )

        left_id = self._build(objects, start, mid)
        right_id = self._build(objects, mid, end)
        self.nodes[used_nodes].aabb = self.nodes[left_id].aabb.union(
            self.nodes[right_id].aabb
        )

        self.nodes[used_nodes].left_id = left_id
        self.nodes[used_nodes].right_id = right_id

        return used_nodes

    def info(self) -> None:
        print(f"BVH Info:")
        print(f"  Number of objects: {len(self.objects)}")
        print(f"  Number of nodes: {self.used_nodes}")
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


def sort_key(obj, axis: int):
    if obj.shape == ObjectShape.TRIANGLE:
        return obj.entity.bbox.min[axis]
    elif obj.shape == ObjectShape.SPHERE:
        return obj.entity.bbox.min[axis] + 2 * TMAX
    else:
        raise ValueError(f"Unsupported object shape: {obj.shape}")


def sort_objects(objects: List, axis: int) -> List:
    return sorted(objects, key=lambda obj: sort_key(obj, axis))
