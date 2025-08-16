from typing import Optional

import numpy as np
import taichi as ti
from taichi.math import vec3

from ..utils import ObjectTag, PBRPreset, load_obj


@ti.data_oriented
class Mesh:
    def __init__(self) -> None:
        pass

    def load(
        self,
        tag: int,
        vertices: np.ndarray,
        indices: Optional[np.ndarray] = None,
        texture_coords: Optional[np.ndarray] = None,
        coords_mapping: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        from ..objects import PBRMaterial

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

        if texture_coords is not None and coords_mapping is not None:
            self.texture_coords = ti.Vector.field(
                texture_coords.shape[1], dtype=ti.f32, shape=texture_coords.shape[0]
            )
            self.texture_coords.from_numpy(texture_coords)
            self.coords_mapping = ti.Vector.field(
                coords_mapping.shape[1], dtype=ti.i32, shape=coords_mapping.shape[0]
            )
            self.coords_mapping.from_numpy(coords_mapping)

        else:
            self.texture_coords = None
            self.coords_mapping = None

        self.kwargs = kwargs
        if self.tag == ObjectTag.PBR:
            self.kwargs = {"pbr": PBRMaterial(**kwargs)}

    def load_file(
        self,
        tag: int,
        obj_file: str,
        **kwargs,
    ) -> None:
        obj = load_obj(obj_file)
        self.load(
            tag,
            obj["vertices"],
            obj["indices"],
            obj["texture_coords"],
            obj["coords_mapping"],
            **kwargs,
        )

    def use(self, preset: PBRPreset) -> None:
        config = preset.value
        self.metallic.get(config["metallic"], 0.0)
        self.roughness.get(config["roughness"], 0.0)
        self.emission.get(config["emission"], vec3(0.0))
