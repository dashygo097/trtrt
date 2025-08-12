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
    coords_mapping = []

    with open(obj_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] == "v":
                vertices.append(list(map(float, tokens[1:4])))
            elif tokens[0] == "vt":
                texture_coords.append(list(map(float, tokens[1:3])))
            elif tokens[0] == "f":
                face_idx = []
                face_uv = []
                for vert in tokens[1:]:
                    if "/" in vert:
                        v_idx, vt_idx = vert.split("/")[:2]
                        vi = int(v_idx) - 1
                        face_idx.append(vi)

                        if vt_idx:
                            face_uv.append(int(vt_idx) - 1)
                        else:
                            face_uv.append(None)
                    else:
                        vi = int(vert) - 1
                        face_idx.append(vi)
                        face_uv.append(None)

                indices.append(face_idx)
                coords_mapping.append(face_uv)

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "indices": np.array(indices, dtype=np.int32),
        "texture_coords": np.array(texture_coords, dtype=np.float32)
        if texture_coords
        else None,
        "coords_mapping": coords_mapping if texture_coords else None,
    }
