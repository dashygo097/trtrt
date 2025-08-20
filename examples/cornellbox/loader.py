from taichi.math import vec3

import trtrt as tr


def load_cornellbox():
    floor = tr.Mesh()
    left = tr.Mesh()
    right = tr.Mesh()
    shortbox = tr.Mesh()
    tallbox = tr.Mesh()
    lit = tr.Mesh()

    floor.from_file("./assets/floor.obj")
    left.from_file("./assets/left.obj")
    right.from_file("./assets/right.obj")
    shortbox.from_file("./assets/shortbox.obj")
    tallbox.from_file("./assets/tallbox.obj")
    lit.from_file("./assets/light.obj")

    floor.set_material(tr.ObjectTag.PBR, albedo=vec3(1.0))
    left.set_material(tr.ObjectTag.PBR, albedo=vec3(1.0, 0.0, 0.0), roughness=1.0)
    right.set_material(tr.ObjectTag.PBR, albedo=vec3(0.0, 1.0, 0.0), roughness=1.0)
    shortbox.set_material(tr.ObjectTag.PBR, albedo=vec3(1.0), roughness=1.0)
    tallbox.set_material(
        tr.ObjectTag.PBR, albedo=vec3(1.0), metallic=1.0, roughness=0.2
    )
    lit.set_material(tr.ObjectTag.PBR, albedo=vec3(1.0), emission=vec3(10.0))

    s = tr.Scene(maximum=512)
    s.add_obj(floor)
    s.add_obj(left)
    s.add_obj(right)
    s.add_obj(shortbox)
    s.add_obj(tallbox)
    s.add_obj(lit)
    s.set_bg(vec3(1.0, 0.1, 0.6))
    c = tr.Camera()
    c.set_lookat(2.50, 3.0, 0.0)
    c.set_lookfrom(2.50, 3.50, -6.0)
    return s, c
