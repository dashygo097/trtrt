import taichi as ti
from loader import load_cornellbox

import trtrt.core as g

ti.init(arch=ti.gpu, debug=True)
res = (800, 800)
s, c = load_cornellbox()
r = g.PathTracer(sampler=g.UniformSampler(), samples_per_pixel=1)


def cornell_main():
    f = g.FrontEnd("Cornell Box", res=res)

    f.set_scene(s)
    f.set_camera(c)
    f.set_renderer(r)
    f.add_post_processor(g.GaussianBlur(enabled=False))
    f.add_post_processor(g.BilateralFilter(enabled=False))
    f.add_post_processor(g.JointBilateralFilter(enabled=False))

    f.run()


if __name__ == "__main__":
    cornell_main()
