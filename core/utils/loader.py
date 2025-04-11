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
    cmap = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.strip().split(" ")[1:]])

            elif line.startswith("f "):
                indices.append([int(x) for x in line.strip().split(" ")[1:]])

            elif line.startswith("c "):
                cmap.append([float(x) for x in line.strip().split(" ")[1:]])
    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "indices": np.array(indices, dtype=np.int32) - 1,
    }
