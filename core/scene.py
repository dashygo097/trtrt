from typing import Dict, List, Optional, Union, overload

import numpy as np
import taichi as ti
from taichi.math import vec3
from termcolor import colored

from .bvh import BVH, Stack, bbox_valid
from .objects import DirecLight, Ray, Sphere, Triangle, init4bbox
from .records import BVHHitInfo, HitInfo
from .utils.const import TMAX, TMIN, ObjectShape, ObjectTag, PBRPreset
from .utils.loader import load_obj


@ti.data_oriented
class Mesh:
    # TODO: Implement the triangle mesh and provide interface for the 'Meshes' cls
    def __init__(self) -> None:
        pass

    def load(
        self,
        tag: int,
        vertices: np.ndarray,
        indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        self.tag = tag
        self.vertices = ti.Vector.field(
            vertices.shape[1], dtype=ti.f32, shape=vertices.shape[0]
        )
        self.vertices.from_numpy(vertices)
        if indices is not None:
            self.indices = ti.Vector.field(
                indices.shape[1], dtype=ti.i32, shape=indices.shape[0]
            )
            self.indices.from_numpy(indices)

        else:
            self.indices = None

        self.kwargs = kwargs
        self.albedo = kwargs.get("albedo", vec3(0.0))
        self.metallic = kwargs.get("metallic", 0.0)
        self.roughness = kwargs.get("roughness", 0.0)
        self.emission = kwargs.get("emission", vec3(0.0))

    def load_file(
        self,
        tag: int,
        obj_file: str,
        **kwargs,
    ) -> None:
        obj = load_obj(obj_file)
        self.load(tag, obj["vertices"], obj["indices"], **kwargs)

    def use(self, preset: PBRPreset) -> None:
        config = preset.value
        self.metallic.get(config["metallic"], 0.0)
        self.roughness.get(config["roughness"], 0.0)
        self.emission.get(config["emission"], vec3(0.0))


@ti.data_oriented
class Meshes:
    def __init__(self, maximum: int = 32, use_bvh: bool = False) -> None:
        self.maximum = ti.field(ti.i32, shape=())
        self.use_bvh = ti.field(ti.u1, shape=())
        self.maximum[None] = maximum * 2
        self.use_bvh[None] = use_bvh

        self.objects = []
        self.label = []

        self.light_ptr = ti.field(dtype=ti.i32, shape=())
        self.light_map = ti.field(dtype=ti.i32, shape=maximum)

        self.mesh = Triangle.field(shape=maximum)
        self.tri_ptr = ti.field(dtype=ti.i32, shape=())

        self.spheres = Sphere.field(shape=maximum)
        self.sphere_ptr = ti.field(dtype=ti.i32, shape=())

        self.light_ptr[None] = 0
        self.tri_ptr[None] = 0
        self.sphere_ptr[None] = 0
        self.bg_top = vec3(0.0)
        self.bg_bottom = vec3(0.0)
        self.dir_light = DirecLight()

        self.bvh = BVH()
        self.stack = Stack(maximum * 4)
        self.hit_count = ti.field(dtype=ti.i32, shape=())

        self.params: Dict = {}
        self.update()

    def __getitem__(self, index: int):
        if 0 <= index < self.tri_ptr[None]:
            obj = self.mesh[index]
        elif self.tri_ptr[None] <= index < self.tri_ptr[None] + self.sphere_ptr[None]:
            obj = self.spheres[index - self.tri_ptr[None]]
        else:
            raise IndexError("Index out of range")

        return obj

    def set_bvh(self, use_bvh: bool) -> None:
        self.use_bvh[None] = use_bvh
        self.update()

    def update(self) -> None:
        self.params["maximum"] = self.maximum[None]
        self.params["use_bvh"] = self.use_bvh[None]

    def get_params(self) -> Dict:
        return {"maximum": self.maximum[None], "use_bvh": self.use_bvh[None]}

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

    def make(self) -> None:
        for i in range(len(self.objects)):
            init4bbox(self.objects[i])

        for index, obj in enumerate(self.objects):
            label = self.label[index]
            if label[0] == ObjectTag.PBR and obj.emission.norm() > 0.0:
                self.light_map[self.light_ptr[None]] = index
                self.light_ptr[None] += 1

            else:
                pass

            if label[1] == ObjectShape.TRIANGLE:
                self.mesh[self.tri_ptr[None]] = obj
                self.tri_ptr[None] += 1

            elif label[1] == ObjectShape.SPHERE:
                self.spheres[self.sphere_ptr[None]] = obj
                self.sphere_ptr[None] += 1

            else:
                pass

        self.bvh.set_objects(self.objects)
        self.bvh.build()

        self.bvh.pretty_print()
        self.bvh.info()
        self.info()

    def info(self) -> None:
        has_directional_light = 1 if self.dir_light.color.norm() > 0.0 else 0
        print("[INFO] BUILD SUCCESS!")
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Triangles", "green", attrs=["bold"])
            + colored(f": {self.tri_ptr[None]}", attrs=["bold"])
        )
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Spheres", "red", attrs=["bold"])
            + colored(f": {self.sphere_ptr[None]}", attrs=["bold"])
        )
        print(
            "[INFO] "
            + colored("Number of ", attrs=["bold"])
            + colored("Light Emitters", "yellow", attrs=["bold"])
            + colored(
                f": {self.light_ptr[None]}",
                attrs=["bold"],
            )
        )
        if has_directional_light:
            print(
                "[INFO] "
                + colored("Has ", attrs=["bold"])
                + colored("Directional Light", "yellow", attrs=["bold"])
            )

    def set_bg(self, color: ti.Vector) -> None:
        self.bg_top = color
        self.bg_bottom = color

    def set_bg_gradient(self, bottom: ti.Vector, top: ti.Vector) -> None:
        self.bg_top = top
        self.bg_bottom = bottom

    def set_dir_light(self, dir_light: DirecLight) -> None:
        self.dir_light = dir_light

    @ti.func
    def bvh_intersect(self, ray, tmin=TMIN, tmax=TMAX) -> BVHHitInfo:
        bvh_hitinfo = BVHHitInfo(is_hit=False, tmin=tmin, tmax=tmax, obj_id=-1)
        hitinfo = HitInfo(time=tmax)

        stack = ti.Vector([-1] * 512, dt=ti.i32)
        stack[0] = self.bvh.root_id
        stack_ptr = 1

        self.hit_count[None] = 0

        for _ in range(512):
            if stack_ptr == 0:
                break

            node_id = stack[stack_ptr - 1]
            stack_ptr -= 1

            if node_id == -1:
                continue

            node = self.bvh.nodes[node_id]
            box_tmin, box_tmax = node.aabb.intersect(ray)
            self.hit_count[None] += 1

            if bbox_valid(box_tmin, box_tmax):
                if node.obj_id != -1:
                    if node.obj_id < self.tri_ptr[None]:
                        obj = self.mesh[node.obj_id]
                        obj_hit = obj.intersect(ray, tmin, hitinfo.time)
                        self.hit_count[None] += 1

                        if obj_hit.is_hit and obj_hit.time < hitinfo.time:
                            hitinfo = obj_hit
                            bvh_hitinfo.is_hit = True
                            bvh_hitinfo.tmin = obj_hit.time
                            bvh_hitinfo.tmax = obj_hit.time
                            bvh_hitinfo.obj_id = node.obj_id

                    elif node.obj_id < self.tri_ptr[None] + self.sphere_ptr[None]:
                        obj = self.spheres[node.obj_id - self.tri_ptr[None]]
                        obj_hit = obj.intersect(ray, tmin, hitinfo.time)
                        self.hit_count[None] += 1

                        if obj_hit.is_hit and obj_hit.time < hitinfo.time:
                            hitinfo = obj_hit
                            bvh_hitinfo.is_hit = True
                            bvh_hitinfo.tmin = obj_hit.time
                            bvh_hitinfo.tmax = obj_hit.time
                            bvh_hitinfo.obj_id = node.obj_id
                else:
                    if node.left_id != -1:
                        stack[stack_ptr] = node.left_id
                        stack_ptr += 1
                    if node.right_id != -1:
                        stack[stack_ptr] = node.right_id
                        stack_ptr += 1

        return bvh_hitinfo, hitinfo

    @ti.func
    def bruteforce_intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        hitinfo = HitInfo(time=tmax)
        hitinfo_tmp = HitInfo(time=tmax)

        for index in range(self.tri_ptr[None]):
            hitinfo_tmp = self.mesh[index].intersect(
                ray,
                tmin,
                hitinfo.time,
            )

            if hitinfo_tmp.is_hit and hitinfo_tmp.time < hitinfo.time:
                hitinfo = hitinfo_tmp

        for index in range(self.sphere_ptr[None]):
            hitinfo_tmp = self.spheres[index].intersect(
                ray,
                tmin,
                hitinfo.time,
            )

            if hitinfo_tmp.is_hit and hitinfo_tmp.time < hitinfo.time:
                hitinfo = hitinfo_tmp

        return hitinfo

    @ti.func
    def intersect(self, ray: Ray, tmin=TMIN, tmax=TMAX) -> HitInfo:
        hitinfo = self.bruteforce_intersect(ray, tmin, tmax)
        return hitinfo

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
            self.label.append((obj.tag, ObjectShape.TRIANGLE))
            self.objects.append(obj)
        elif len(args) == 1 and isinstance(args[0], Sphere):
            obj = args[0]
            self.label.append((obj.tag, ObjectShape.SPHERE))
            self.objects.append(obj)
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
