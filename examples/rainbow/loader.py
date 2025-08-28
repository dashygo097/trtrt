import numpy as np
from taichi.math import vec3

import trtrt as tr


def load_rainbow():
    floor = tr.Mesh()
    blank = tr.Mesh()
    room = tr.Mesh()
    rainbow = [tr.Mesh() for _ in range(7)]

    rainbow_vertices = np.array(
        [[-7, 0.01, -4], [-5, 0.01, -4], [-5, 0.01, 4], [-7, 0.01, 4]],
        dtype=np.float32,
    )

    rainbow_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    for i in range(7):
        rainbow[i].load_geometry(
            rainbow_vertices + np.array([[2 * i, 0, 0]] * 4, dtype=np.float32),
            rainbow_indices,
        )
        color = vec3(0.0)
        if i == 0:
            color = vec3(1.0, 0.0, 0.0)
        elif i == 1:
            color = vec3(1.0, 0.5, 0.0)
        elif i == 2:
            color = vec3(1.0, 1.0, 0.0)
        elif i == 3:
            color = vec3(0.0, 1.0, 0.0)
        elif i == 4:
            color = vec3(0.0, 1.0, 1.0)
        elif i == 5:
            color = vec3(0.0, 0.0, 1.0)
        elif i == 6:
            color = vec3(0.5, 0.0, 1.0)

        rainbow[i].set_material(
            tr.ObjectTag.PBR,
            albedo=color,
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
            [[-7, 0, 4], [7, 0, 4], [7, 4, 4], [-7, 4, 4]],
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
                [-7, 0, -4],
                [7, 0, -4],
                [7, 0, 4],
                [-7, 0, 4],
                [-7, 6, -4],
                [7, 6, -4],
                [7, 6, 4],
                [-7, 6, 4],
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

    dlight = tr.DirecLight(dir=vec3(0.5, -2.0, -2.0), color=vec3(10.0))
    alight = tr.AmbientLight(color=vec3(10.0))

    s = tr.Scene(maximum=5000)
    s.add_obj([floor, blank, room])
    s.add_obj(rainbow)
    s.set_directional_light(dlight)
    s.set_ambient_light(alight)
    c = tr.Camera()
    c.set_lookfrom(0, 4, 8)
    c.set_lookat(0, 1, -1)

    return s, c
