import os
import sys
from typing import Dict

import numpy as np


def load_obj(obj_path: str) -> Dict:
    """
    Load a .obj file and return the vertices as a numpy array.
    """

    dirname = sys.path[0]
    obj_path = os.path.join(dirname, obj_path)
    obj_path = os.path.abspath(obj_path)

    vertices = []
    indices = []
    texture_coords = []
    coords_mapping = {}

    has_texture = False

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("vt "):
                has_texture = True
                texture_coords.append([float(x) for x in line.strip().split(" ")[1:]])

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.strip().split(" ")[1:]])

            elif line.startswith("f "):
                if has_texture:
                    indices.append(
                        [int(x.split("/")[0]) for x in line.strip().split(" ")[1:]]
                    )
                    coords_mapping.update(
                        {
                            int(x.split("/")[0]): int(x.split("/")[1])
                            for x in line.strip().split(" ")[1:]
                        }
                    )
                else:
                    indices.append([int(x) for x in line.strip().split(" ")[1:]])

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "indices": np.array(indices, dtype=np.int32) - 1 if indices else None,
        "texture_coords": (
            np.array(texture_coords, dtype=np.float32) if has_texture else None
        ),
        "coords_mapping": coords_mapping if has_texture else None,
    }
