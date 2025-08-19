from typing import Dict


class UIBuilder:
    _renderer_ui: Dict = {}
    _post_ui: Dict = {}

    def __init__(self, frontend) -> None:
        self.frontend = frontend

    @classmethod
    def register_renderer(cls, target_cls):
        def decorator(func):
            cls._renderer_ui[target_cls] = func
            return func

        return decorator

    @classmethod
    def register_post(cls, target_cls):
        def decorator(func):
            cls._post_ui[target_cls] = func
            return func

        return decorator

    def render(self) -> None:
        self.frontend.panel_update = False

        if self.frontend.input_tracer.is_showing_panel():
            # Renderer
            with self.frontend.gui.sub_window("Renderer", 0.02, 0.02, 0.3, 0.2):
                self.frontend.gui.text(self.frontend.renderer._name())
                self.frontend.gui.text(
                    f"Sampler: {self.frontend.renderer.params['sampler']}"
                )
                spp = self.frontend.gui.slider_int(
                    "spp", self.frontend.renderer.params["samples_per_pixel"], 1, 10
                )
                if self.frontend.renderer.params["samples_per_pixel"] != spp:
                    self.frontend.renderer.set_spp(spp)
                    self.frontend.panel_update = True

                for cls_type, func in self._renderer_ui.items():
                    if isinstance(self.frontend.renderer, cls_type):
                        func(self.frontend)

            # Camera
            with self.frontend.gui.sub_window("Camera", 0.02, 0.22, 0.3, 0.08):
                fov = self.frontend.gui.slider_float(
                    "fov", self.frontend.camera.params["fov"], 30.0, 150.0
                )
                if self.frontend.camera.params["fov"] != fov:
                    self.frontend.camera.set_fov(fov)
                    self.frontend.panel_update = True

            # Post Process
            for index, core in enumerate(self.frontend.post_processors):
                with self.frontend.gui.sub_window(
                    f"Post Processor Core_{index}", 0.02, 0.3 + 0.12 * index, 0.3, 0.12
                ):
                    self.frontend.gui.text(core._name())
                    click_enable = self.frontend.gui.button(
                        f"{'Enabled' if core.params['enabled'] else 'Disabled'}",
                    )
                    if click_enable:
                        core.toggle()
                    for cls_type, func in self._post_ui.items():
                        if isinstance(core, cls_type):
                            func(self.frontend, core)

            # Control Panel
            with self.frontend.gui.sub_window("Control Panel", 0.02, 0.85, 0.2, 0.12):
                max_fps = self.frontend.gui.slider_int(
                    "Target FPS", self.frontend.fps[None], 10, 90
                )
                panel_update = self.frontend.gui.button("Refresh")
                if panel_update:
                    self.frontend.panel_update = True
                if self.frontend.fps[None] != max_fps:
                    self.frontend.fps[None] = max_fps
                    self.frontend._set_fps(max_fps)
                    self.frontend.panel_update = True
