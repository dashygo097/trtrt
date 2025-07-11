from typing import List, Union, overload

import numpy as np
import taichi as ti
from taichi.math import vec3
from termcolor import colored

from .geometry.bvh import BVH
from .geometry.mesh import Mesh
from .objects import DirecLight, Ray, Sphere, Triangle, init4bbox
from .records import BVHHitInfo, HitInfo
from .utils.abstract import Abstraction
from .utils.const import TMAX, TMIN, ObjectShape, ObjectTag


@ti.data_oriented
class Scene:
    def __init__(self, maximum: int = 100) -> None:
        self.maximum = maximum

        self.objects: List[Abstraction] = []

        self.light_map = ti.field(dtype=ti.i32, shape=maximum)
        self.light_ptr = 0

        self.mesh = Triangle.field(shape=maximum)
        self.tri_ptr = 0

        self.spheres = Sphere.field(shape=maximum)
        self.sphere_ptr = 0

        self.bg_top = vec3(0.0)
        self.bg_bottom = vec3(0.0)

        self.dir_light = DirecLight()

        self.bvh = BVH()

        # self.hit_count = ti.field(dtype=ti.i32, shape=())
        # self.hit_count[None] = 0

    def __getitem__(self, index: int):
        if 0 <= index < self.tri_ptr:
            obj = self.mesh[index]
        elif self.tri_ptr <= index < self.tri_ptr + self.sphere_ptr:
            obj = self.spheres[index - self.tri_ptr]
        else:
            raise IndexError("Index out of range")

        return obj

    @overload
    def add_mesh(self, mesh: Mesh) -> None:
        # TODO: If the indices buffer is not provided, implement the delaunay triangulation
        self.add_mesh(mesh)

    @overload
    def add_mesh(
        self,
        tag: ObjectTag,
        vertices: np.ndarray,
        indices: np.ndarray,
        **kwargs,
    ) -> None:
        self.add_mesh(tag, vertices, indices, **kwargs)

    @overload
    def add_mesh(
        self,
        tag: int,
        vertices: np.ndarray,
        indices: np.ndarray,
        **kwargs,
    ) -> None:
        self.add_mesh(tag, vertices, indices, **kwargs)

    @overload
    def add_obj(
        self,
        obj: Union[Triangle, Sphere, Mesh],
    ) -> None:
        self.add_obj(obj)

    @overload
    def add_obj(self, objs: List) -> None:
        self.add_obj(objs)

    def set_bg(self, color: ti.Vector) -> None:
        self.bg_top = color
        self.bg_bottom = color

    def set_bg_gradient(self, bottom: ti.Vector, top: ti.Vector) -> None:
        self.bg_top = top
        self.bg_bottom = bottom

    def set_dir_light(self, dir_light: DirecLight) -> None:
        self.dir_light = dir_light

    @ti.func
    def bvh_intersect(self, ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        hitinfo = HitInfo(time=tmax)
        hitinfo_tmp = HitInfo(time=tmax)

        stack = ti.Vector([-1] * 2 * (self.tri_ptr + self.sphere_ptr), dt=ti.i32)
        stack[0] = self.bvh.root_id
        stack_ptr = 1

        # self.hit_count[None] = 0

        while stack_ptr > 0:
            stack_ptr -= 1
            node_id = stack[stack_ptr]
            stack[stack_ptr] = -1

            if node_id == -1:
                continue

            node = self.bvh.nodes[node_id]

            aabb_hit = node.aabb.intersect(ray, tmin, hitinfo.time)
            # self.hit_count[None] += 1
            if not aabb_hit.is_hit or aabb_hit.tmin >= hitinfo.time:
                continue

            if node.obj_id != -1:
                if node.obj_id < self.tri_ptr:
                    hitinfo_tmp = self.mesh[node.obj_id].intersect(
                        ray, tmin, hitinfo.time
                    )
                    # self.hit_count[None] += 1
                elif node.obj_id < self.tri_ptr + self.sphere_ptr:
                    hitinfo_tmp = self.spheres[node.obj_id - self.tri_ptr].intersect(
                        ray, tmin, hitinfo.time
                    )
                    # self.hit_count[None] += 1

                if hitinfo_tmp.is_hit and (hitinfo_tmp.time < hitinfo.time):
                    hitinfo = hitinfo_tmp
            else:
                if node.right_id != -1:
                    stack[stack_ptr] = node.right_id
                    stack_ptr += 1
                if node.left_id != -1:
                    stack[stack_ptr] = node.left_id
                    stack_ptr += 1

        return hitinfo

    @ti.func
    def bruteforce_intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        hitinfo = HitInfo(time=tmax)
        hitinfo_tmp = HitInfo(time=tmax)

        # self.hit_count[None] = 0

        for index in range(self.tri_ptr):
            hitinfo_tmp = self.mesh[index].intersect(ray, tmin, hitinfo.time)
            # self.hit_count[None] += 1

            if hitinfo_tmp.is_hit and (hitinfo_tmp.time < hitinfo.time):
                hitinfo = hitinfo_tmp

        for index in range(self.sphere_ptr):
            hitinfo_tmp = self.spheres[index].intersect(ray, tmin, hitinfo.time)
            # self.hit_count[None] += 1

            if hitinfo_tmp.is_hit and (hitinfo_tmp.time < hitinfo.time):
                hitinfo = hitinfo_tmp

        return hitinfo

    @ti.func
    def intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        return self.bruteforce_intersect(ray, tmin, tmax)

    def make(self) -> None:
        for i in range(len(self.objects)):
            init4bbox(self.objects[i].entity)

        self.bvh.set_objects(self.objects)
        self.bvh.build()

        self.bvh.pretty_print()
        self.bvh.info()

        for index, obj in enumerate(self.objects):
            if obj.tag == ObjectTag.PBR and obj.entity.pbr.emission.norm() > 0.0:
                self.light_map[self.light_ptr] = index
                self.light_ptr += 1

            if obj.shape == ObjectShape.TRIANGLE:
                self.mesh[self.tri_ptr] = obj.entity
                self.tri_ptr += 1
            elif obj.shape == ObjectShape.SPHERE:
                self.spheres[self.sphere_ptr] = obj.entity
                self.sphere_ptr += 1

        self.info()

    def info(self) -> None:
        has_directional_light = 1 if self.dir_light.color.norm() > 0.0 else 0
        print("[INFO] BUILD SUCCESS!")
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Triangles", "green", attrs=["bold"])
            + colored(f": {self.tri_ptr}", attrs=["bold"])
        )
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Spheres", "red", attrs=["bold"])
            + colored(f": {self.sphere_ptr}", attrs=["bold"])
        )
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Light Emitters", "yellow", attrs=["bold"])
            + colored(
                f": {self.light_ptr}",
                attrs=["bold"],
            )
        )
        if has_directional_light:
            print(
                "[INFO] "
                + colored("Has ", attrs=["bold"])
                + colored("Directional Light", "yellow", attrs=["bold"])
            )

    def add_mesh(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], Mesh):
            mesh = args[0]
            assert mesh.indices is not None, "Indices buffer is required"

            vertices = mesh.vertices.to_numpy()
            indices = mesh.indices.to_numpy()
            self.add_mesh(mesh.tag, vertices, indices, **mesh.kwargs)

        elif (
            len(args) == 3
            and isinstance(args[0], int)
            and isinstance(args[1], np.ndarray)
            and isinstance(args[2], np.ndarray)
        ):
            tag = args[0]
            vertices = args[1]
            indices = args[2]
            n_tris = indices.shape[0]
            for index in range(n_tris):
                v0 = vertices[indices[index, 0]]
                v1 = vertices[indices[index, 1]]
                v2 = vertices[indices[index, 2]]
                self.add_obj(Triangle(tag=tag, v0=v0, v1=v1, v2=v2, **kwargs))

        else:
            raise ValueError(
                "Invalid arguments, please provide either a Mesh object or (tag, vertices, indices, **kwargs)"
            )

    def add_obj(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], Triangle):
            obj = args[0]
            self.objects.append(Abstraction(obj))
        elif len(args) == 1 and isinstance(args[0], Sphere):
            obj = args[0]
            self.objects.append(Abstraction(obj))
        elif len(args) == 1 and isinstance(args[0], Mesh):
            obj = args[0]
            self.add_mesh(obj)
        elif len(args) == 1 and isinstance(args[0], List):
            for obj in args[0]:
                self.add_obj(obj)

        else:
            raise ValueError(
                "Invalid arguments, please provide either a Triangle, Sphere, or Mesh object"
            )
