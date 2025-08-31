import time
from typing import List, Tuple

import taichi as ti
from taichi.math import vec3
from termcolor import colored

from .camera import Camera
from .objects import Ray
from .postprocess import JointBilateralFilter, ProcessorCore, ToneMapping
from .records import GBuffer, VelocityBuffer
from .renderer import Albedo, Renderer
from .scene import Scene
from .ui import InputTracer, UIBuilder


@ti.data_oriented
class FrontEnd:
    def __init__(self, name: str, res: Tuple[int, int]) -> None:
        self.res = res

        # Buffers
        self.tmp_buffers = ti.Vector.field(3, dtype=ti.f32, shape=res)
        self.acc_buffers = ti.Vector.field(3, dtype=ti.f32, shape=res)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)

        # Components
        self.window = ti.ui.Window(name, res)
        self.gui = self.window.get_gui()
        self.input_tracer = InputTracer(self.window, pixels=self.pixels)
        self.camera = Camera()
        self.scene = Scene()
        self.renderer = Albedo()
        self.post_processors: List[ProcessorCore] = []

        # State
        self.panel_update: bool = False
        self.cnt = ti.field(dtype=ti.i32, shape=())
        self.fps = ti.field(dtype=ti.i32, shape=())
        self.fps[None] = 45

        # Materials
        self.g_buffer = GBuffer.field(shape=res)
        self.velocity_buffer = VelocityBuffer(self.res, self.camera)

        # Post Processors
        self.add_post_processor(ToneMapping(enabled=True))

        # UI
        self.ui = UIBuilder(self)

        self.prev_camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

    def _set_fps(self, fps: float) -> None:
        self.fps[None] = fps

    def set_tonemap(self, tonemap: ToneMapping) -> None:
        self.post_processors[0] = tonemap
        self.post_processors[0].set_buffers(self.tmp_buffers)

    def set_camera(self, camera: Camera) -> None:
        self.camera = camera
        self.velocity_buffer = VelocityBuffer(self.res, self.camera)

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene
        self.scene.make()

    def set_renderer(self, renderer: Renderer) -> None:
        self.renderer = renderer

    def add_post_processor(self, post_processor: ProcessorCore) -> None:
        if len(self.post_processors) == 0:
            post_processor.set_buffers(self.tmp_buffers)
        else:
            post_processor.set_buffers(self.post_processors[-1].buffers)

        self.post_processors.append(post_processor)

    def info(self) -> None:
        print(
            "[INFO] "
            + colored("Renderer: ", attrs=["bold"])
            + colored(f"{self.renderer._name()}", color="blue", attrs=["bold"])
        )

    @ti.kernel
    def clear(self):
        self.acc_buffers.fill(0)

    @ti.kernel
    def render(self):
        ray = Ray()
        for i, j in self.tmp_buffers:
            color = vec3(0.0)
            u = (i + ti.random()) / self.res[0]
            v = (j + ti.random()) / self.res[1]
            for _ in range(self.renderer.samples_per_pixel[None]):
                ray = self.camera.get_ray(u, v)
                color += self.renderer.ray_color(self.scene, ray, u, v)
            self.tmp_buffers[i, j] = color / self.renderer.samples_per_pixel[None]
            self.g_buffer[i, j] = self.renderer.fetch_gbuffer(self.scene, ray)

    @ti.kernel
    def accumulate(self):
        for i, j in self.acc_buffers:
            self.acc_buffers[i, j] += self.tmp_buffers[i, j]
            # Gamma correction
            self.pixels[i, j] = ti.sqrt(self.acc_buffers[i, j] / self.cnt[None])

    def _preprocess(self) -> None:
        for core in self.post_processors:
            if isinstance(core, JointBilateralFilter) and core.params["enabled"]:
                core.fetch_gbuffer(self.g_buffer)

    def run(self) -> None:
        self.info()
        canvas = self.window.get_canvas()

        while self.window.running:
            # Control Flow
            t = time.time()
            self.ui.render()
            self.input_tracer.control_panel()
            self.input_tracer.keymap()
            if not self.input_tracer.is_showing_panel():
                self.camera.track(self.window)

            # Taichi Scope
            if self.input_tracer.should_clear() | self.panel_update:
                self.cnt[None] = 1
                self.clear()

            self.render()

            self._preprocess()
            for core in self.post_processors:
                if core.params["enabled"]:
                    core.process()

            self.tmp_buffers = self.post_processors[-1].buffers
            self.accumulate()

            # NOTE: VELOCITY BUFFER TEST
            self.velocity_buffer.store_positions(self.g_buffer)
            self.velocity_buffer.compute_velocity()

            self.input_tracer.pixels = self.pixels
            canvas.set_image(self.pixels)
            self.window.show()

            self.cnt[None] += 1
            time_elapsed = time.time() - t
            if time_elapsed * self.fps[None] < 1.0:
                time.sleep(1 / self.fps[None] - time_elapsed)

            # NOTE: Profiler
            # ti.profiler.print_kernel_profiler_info("trace")
            # ti.profiler.clear_kernel_profiler_info()
            # print(self.scene.hit_count[None])
