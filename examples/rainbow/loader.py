import numpy as np
from taichi.math import vec3

import trtrt as tr


def load_rainbow():
    floor = tr.Mesh()
    blank = tr.Mesh()
    room = tr.Mesh()

    floor.load_geometry(
        np.array(
            [[-100, 0, -100], [100, 0, -100], [100, 0, 100], [-100, 0, 100]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    floor.set_material(
        tr.ObjectTag.PBR,
        albedo=vec3(0.8),
        roughness=1.0,
    )
    blank.load_geometry(
        np.array(
            [[-2.4, 0, 3], [3, 0, 3], [3, 6, 3], [-2.4, 6, 3]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    blank.set_material(
        tr.ObjectTag.PBR,
        albedo=vec3(0.8),
        roughness=1.0,
    )
    room.load_geometry(
        np.array(
            [
                [-3, 0, -3],
                [3, 0, -3],
                [3, 0, 3],
                [-3, 0, 3],
                [-3, 6, -3],
                [3, 6, -3],
                [3, 6, 3],
                [-3, 6, 3],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        ),
    )
    room.set_material(
        tr.ObjectTag.PBR,
        albedo=vec3(0.8),
        roughness=1.0,
    )

    dlight = tr.DirecLight(
        dir=vec3(0.1, -2.0, -1.0),
        color=vec3(1.0),
    )

    s = tr.Scene(maximum=5000)
    s.set_bg(vec3(0.0, 0.4, 1.0))
    s.add_obj([floor, blank, room])
    s.set_dir_light(dlight)
    c = tr.Camera()
    c.set_lookfrom(0, 4, 8)
    c.set_lookat(0, 1, -1)

    return s, c
