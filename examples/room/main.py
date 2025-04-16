import taichi as ti
from loader import load_room

import trtrt.core as g

ti.init(arch=ti.gpu)
res = (800, 800)

s, c = load_room()
r = g.PathTracer(sampler=g.BlueNoiseSampler(), samples_per_pixel=16, max_depth=3)


def room_main():
    f = g.FrontEnd("Room", res=res)

    f.set_scene(s)
    f.set_camera(c)
    f.set_renderer(r)
    f.add_post_processor(g.GaussianBlur(enabled=False))
    f.add_post_processor(g.BilateralFilter(enabled=False))
    f.add_post_processor(g.JointBilateralFilter(enabled=False))

    f.run()


if __name__ == "__main__":
    room_main()
