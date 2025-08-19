from typing import Any, Dict, Optional

import numpy as np

from ..utils import ObjectTag, PBRPreset, load_obj
from .geometry_data import GeometryData


class MaterialFactory:
    @staticmethod
    def create_material(tag: int, **kwargs) -> Any:
        if tag == ObjectTag.PBR:
            from ..objects import PBRMaterial

            return PBRMaterial(**kwargs)
        elif tag == ObjectTag.GLASS:
            from ..objects import GlassMaterial

            return GlassMaterial(**kwargs)


class Mesh:
    def __init__(self) -> None:
        self._tag: Optional[int] = None
        self._geometry: Optional[GeometryData] = None
        self._material: Optional[Any] = None
        self._material_params: Dict[str, Any] = {}

    @property
    def tag(self) -> Optional[int]:
        return self._tag

    @property
    def geometry(self) -> Optional[GeometryData]:
        return self._geometry

    @property
    def material(self) -> Optional[Any]:
        return self._material

    def is_valid(self) -> bool:
        return (
            self.tag is not None
            and self.geometry is not None
            and self.material is not None
        )

    def load_geometry(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        texture_coords: Optional[np.ndarray] = None,
        coords_mapping: Optional[np.ndarray] = None,
    ) -> "Mesh":
        try:
            self._geometry = GeometryData(
                vertices=vertices,
                indices=indices,
                texture_coords=texture_coords,
                coords_mapping=coords_mapping,
            )
            return self
        except ValueError as e:
            raise ValueError(
                f"Failed to load geometry: {e}. Ensure vertices and indices are valid."
            ) from e

    def set_material(self, tag: int, **material_params) -> "Mesh":
        self._tag = tag
        self._material_params = material_params

        try:
            self._material = MaterialFactory.create_material(tag, **material_params)
        except Exception as e:
            raise ValueError(
                f"Failed to create material for tag {tag}: {e}. "
                "Ensure the material parameters are valid."
            ) from e

        return self

    def from_file(self, obj_file: str) -> "Mesh":
        try:
            obj_data = load_obj(obj_file)
            self.load_geometry(
                vertices=obj_data["vertices"],
                indices=obj_data["indices"],
                texture_coords=obj_data.get("texture_coords"),
                coords_mapping=obj_data.get("coords_mapping"),
            )
            return self

        except Exception as e:
            raise IOError(
                f"Failed to load object file {obj_file}: {e}. "
                "Ensure the file exists and is a valid OBJ file."
            ) from e

    def apply_preset(self, preset: PBRPreset) -> "Mesh":
        if self._tag != ObjectTag.PBR:
            raise ValueError("Preset can only be applied to PBR materials.")

        config = preset.value
        if hasattr(self._material, "metallic"):
            self._material.metallic = config.get("metallic", 0.0)
        if hasattr(self._material, "roughness"):
            self._material.roughness = config.get("roughness", 1.0)
        if hasattr(self._material, "emission"):
            emission = config.get("emission", (0.0, 0.0, 0.0))
            self._material.emission = np.array(emission, dtype=np.float32)

        return self
