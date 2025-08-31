import taichi as ti
from loader import load_cornellbox

import trtrt as tr

ti.init(arch=ti.vulkan)
res = (800, 800)
s, c = load_cornellbox()
r = tr.PathTracer(sampler=tr.BlueNoiseSampler(), samples_per_pixel=1)


def cornell_main():
    f = tr.FrontEnd("Cornell Box", res=res)

    f.set_scene(s)
    f.set_camera(c)
    f.set_renderer(r)
    f.add_post_processor(tr.GaussianBlur(enabled=True))
    f.add_post_processor(tr.BilateralFilter(enabled=False))
    f.add_post_processor(tr.JointBilateralFilter(enabled=False))

    f.run()


if __name__ == "__main__":
    cornell_main()
