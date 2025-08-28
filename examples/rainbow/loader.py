import numpy as np
from taichi.math import vec3

import trtrt as tr


def load_rainbow():
    floor = tr.Mesh()
    blank = tr.Mesh()
    room = tr.Mesh()
    rainbow = [tr.Mesh()] * 7

    rainbow_vertices = np.array(
        [[-7, 0.01, -2], [-5, 0.01, -2], [-5, 0.01, 2], [-7, 0.01, 2]],
        dtype=np.float32,
    )

    rainbow_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    for i in range(7):
        rainbow[i].load_geometry(
            rainbow_vertices
            + np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=np.float32),
            rainbow_indices,
        )
        rainbow[i].set_material(
            tr.ObjectTag.PBR,
            albedo=vec3(1.0, i / 6.0, 0.0)
            if i < 3
            else vec3((6 - i) / 6.0, 1.0, 0.0)
            if i < 5
            else vec3(0.0, (i - 4) / 2.0, 1.0),
            roughness=1.0,
        )

    floor.load_geometry(
        np.array(
            [[-100, 0, -100], [100, 0, -100], [100, 0, 100], [-100, 0, 100]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    floor.set_material(
        tr.ObjectTag.PBR,
        albedo=vec3(0.3),
        roughness=1.0,
    )
    blank.load_geometry(
        np.array(
            [[-7, 0, 2], [7, 0, 2], [7, 4, 2], [-7, 4, 2]],
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
                [-7.01, 0, -2.01],
                [7.01, 0, -2.01],
                [7.01, 0, 2.01],
                [-7.01, 0, 2.01],
                [-7.01, 6.01, -2.01],
                [7.01, 6.01, -2.01],
                [7.01, 6.01, 2.01],
                [-7.01, 6.01, 2.01],
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
        dir=vec3(0.5, -1.0, -2.0),
        color=vec3(1.0),
    )

    s = tr.Scene(maximum=5000)
    s.set_bg(vec3(0.2, 0.5, 1.0))
    s.add_obj([floor, blank, room])
    s.add_obj(rainbow)
    s.set_dir_light(dlight)
    c = tr.Camera()
    c.set_lookfrom(0, 4, 8)
    c.set_lookat(0, 1, -1)

    return s, c
