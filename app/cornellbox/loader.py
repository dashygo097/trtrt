from taichi.math import vec3

import trtrt.core as g


def load_cornellbox():
    floor = g.Mesh()
    left = g.Mesh()
    right = g.Mesh()
    shortbox = g.Mesh()
    tallbox = g.Mesh()
    lit = g.Mesh()
    floor.load_file(
        g.ObjectTag.PBR,
        "./assets/floor.obj",
        albedo=vec3(1.0),
    )
    left.load_file(
        g.ObjectTag.PBR,
        "./assets/left.obj",
        albedo=vec3(1.0, 0.0, 0.0),
        roughness=1.0,
    )
    right.load_file(
        g.ObjectTag.PBR,
        "./assets/right.obj",
        albedo=vec3(0.0, 1.0, 0.0),
        roughness=1.0,
    )
    shortbox.load_file(
        g.ObjectTag.PBR,
        "./assets/shortbox.obj",
        albedo=vec3(1.0),
        roughness=1.0,
    )
    tallbox.load_file(
        g.ObjectTag.PBR,
        "./assets/tallbox.obj",
        albedo=vec3(1.0),
        metallic=1.0,
        roughness=0.2,
    )
    lit.load_file(
        g.ObjectTag.PBR,
        "./assets/light.obj",
        albedo=vec3(1.0),
        emission=vec3(10.0),
    )

    s = g.Meshes()
    s.add_obj(floor)
    s.add_obj(left)
    s.add_obj(right)
    s.add_obj(shortbox)
    s.add_obj(tallbox)
    s.add_obj(lit)
    s.set_bg(vec3(0.1, 0.4, 1.0))
    c = g.Camera()
    c.set_lookat(2.50, 3.0, 0.0)
    c.set_lookfrom(2.50, 3.50, -6.0)
    return s, c
