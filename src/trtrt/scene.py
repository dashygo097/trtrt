import os
from typing import List, Union, overload

import numpy as np
import taichi as ti
from termcolor import colored

from .geometry.bvh import BVH
from .geometry.geometry_data import GeometryData
from .geometry.mesh import Mesh
from .objects import AmbientLight, DirecLight, Ray, Sphere, Triangle, init4bbox
from .records import BVHHitInfo, HitInfo
from .utils.abstract import Abstraction
from .utils.const import TMAX, TMIN, ObjectShape, ObjectTag


@ti.data_oriented
class Scene:
    def __init__(self, maximum: int = 128) -> None:
        self.maximum = maximum

        self.objects: List[Abstraction] = []

        self.light_map = ti.field(dtype=ti.i32, shape=maximum)
        self.light_ptr = 0

        self.triangles = Triangle.field(shape=maximum)
        self.tri_ptr = 0

        self.spheres = Sphere.field(shape=maximum)
        self.sphere_ptr = 0

        self.ambient_light = AmbientLight()
        self.directional_light = DirecLight()

        self.bvh = BVH()

    def __getitem__(self, index: int):
        if 0 <= index < self.tri_ptr:
            obj = self.triangles[index]
        elif self.tri_ptr <= index < self.tri_ptr + self.sphere_ptr:
            obj = self.spheres[index - self.tri_ptr]
        else:
            raise IndexError("Index out of range")

        return obj

    @overload
    def add_obj(
        self,
        obj: Union[Triangle, Sphere, Mesh],
    ) -> None:
        self.add_obj(obj)

    @overload
    def add_obj(self, objs: List) -> None:
        self.add_obj(objs)

    @overload
    def add_mesh(self, mesh: Mesh) -> None:
        # TODO: If the indices buffer is not provided, implement the delaunay triangulation
        self.add_mesh(mesh)

    @overload
    def add_mesh(
        self,
        tag: int,
        geometry_data: GeometryData,
        **kwargs,
    ) -> None:
        self.add_mesh(tag, geometry_data, **kwargs)

    @overload
    def add_mesh(
        self,
        tag: int,
        vertices: np.ndarray,
        indices: np.ndarray,
        **kwargs,
    ) -> None:
        self.add_mesh(tag, vertices, indices, **kwargs)

    def set_ambient_light(self, ambient_light: AmbientLight) -> None:
        self.ambient_light = ambient_light

    def set_directional_light(self, directional_light: DirecLight) -> None:
        self.directional_light = directional_light

    def save_meshes(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("# OBJ file\n")
            vertex_offset = 1
            for index in range(self.tri_ptr):
                tri = self.triangles[index]
                v0 = tri.v0
                v1 = tri.v1
                v2 = tri.v2
                f.write(f"v {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"v {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"v {v2[0]} {v2[1]} {v2[2]}\n")
                f.write(f"f {vertex_offset} {vertex_offset + 1} {vertex_offset + 2}\n")
                vertex_offset += 3

    @ti.func
    def bvh_intersect(self, ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        hitinfo = HitInfo(time=tmax)
        hitinfo_tmp = HitInfo(time=tmax)

        stack = ti.Vector([-1] * 2 * (self.tri_ptr + self.sphere_ptr), dt=ti.i32)
        stack[0] = self.bvh.root_id
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node_id = stack[stack_ptr]
            stack[stack_ptr] = -1

            if node_id == -1:
                continue

            node = self.bvh.nodes[node_id]

            aabb_hit = node.aabb.intersect(ray, tmin, hitinfo.time)
            if not aabb_hit.is_hit or aabb_hit.tmin >= hitinfo.time:
                continue

            if node.obj_id != -1:
                if node.obj_id < self.tri_ptr:
                    hitinfo_tmp = self.triangles[node.obj_id].intersect(
                        ray, tmin, hitinfo.time
                    )
                elif node.obj_id < self.tri_ptr + self.sphere_ptr:
                    hitinfo_tmp = self.spheres[node.obj_id - self.tri_ptr].intersect(
                        ray, tmin, hitinfo.time
                    )

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

        for index in range(self.tri_ptr):
            hitinfo_tmp = self.triangles[index].intersect(ray, tmin, hitinfo.time)

            if hitinfo_tmp.is_hit and (hitinfo_tmp.time < hitinfo.time):
                hitinfo = hitinfo_tmp

        for index in range(self.sphere_ptr):
            hitinfo_tmp = self.spheres[index].intersect(ray, tmin, hitinfo.time)

            if hitinfo_tmp.is_hit and (hitinfo_tmp.time < hitinfo.time):
                hitinfo = hitinfo_tmp

        return hitinfo

    @ti.func
    def intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        return self.bruteforce_intersect(ray, tmin, tmax)

    def make(self, bvh_info: bool = False) -> None:
        self.bvh.set_objects(self.objects)
        self.bvh.build()

        if bvh_info:
            self.bvh.pretty_print()
            self.bvh.info()

        for index, obj in enumerate(self.objects):
            if obj.tag == ObjectTag.PBR and obj.entity.emission.norm() > 0.0:
                self.light_map[self.light_ptr] = index
                self.light_ptr += 1

            if obj.shape == ObjectShape.TRIANGLE:
                self.triangles[self.tri_ptr] = obj.entity
                self.tri_ptr += 1
            elif obj.shape == ObjectShape.SPHERE:
                self.spheres[self.sphere_ptr] = obj.entity
                self.sphere_ptr += 1

        self.info()

    def info(self) -> None:
        has_directional_light = 1 if self.directional_light.color.max() > 0.0 else 0
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

    def add_mesh(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], Mesh):
            mesh = args[0]
            self._add_mesh(mesh)
        elif len(args) >= 2 and isinstance(args[1], GeometryData):
            tag = args[0]
            geometry_data = args[1]
            self._add_mesh_from_geometry_data(tag, geometry_data, **kwargs)
        elif (
            len(args) >= 3
            and isinstance(args[1], np.ndarray)
            and isinstance(args[2], np.ndarray)
        ):
            tag = args[0]
            vertices = args[1]
            indices = args[2]
            self._add_mesh_from_arrays(tag, vertices, indices, **kwargs)
        else:
            raise TypeError(f"Invalid arguments for add_mesh: {args} {kwargs}")

    def _add_mesh(self, mesh: Mesh) -> None:
        if mesh.geometry is not None and mesh.tag is not None:
            self._add_mesh_from_arrays(
                mesh.tag,
                mesh.geometry.vertices,
                mesh.geometry.indices,
                **mesh._material_params,
            )

    def _add_mesh_from_geometry_data(
        self, tag: int, geometry_data: GeometryData, **kwargs
    ) -> None:
        self._add_mesh_from_arrays(
            tag,
            geometry_data.vertices,
            geometry_data.indices,
            **kwargs,
        )

    def _add_mesh_from_arrays(
        self, tag: int, vertices: np.ndarray, indices: np.ndarray, **kwargs
    ) -> None:
        n_tris = indices.shape[0]
        for index in range(n_tris):
            v0 = vertices[indices[index, 0]]
            v1 = vertices[indices[index, 1]]
            v2 = vertices[indices[index, 2]]
            obj = Triangle(tag=tag, v0=v0, v1=v1, v2=v2, **kwargs)
            init4bbox(obj)
            self.add_obj(obj)
