from ..postprocess import (BilateralFilter, GaussianBlur, JointBilateralFilter,
                           ToneMapping)
from ..renderer import BlinnPhong, PathTracer, ZBuffer


def make_ui(f) -> None:
    f.panel_update = False

    if f.input_tracer.is_showing_panel():
        # Renderer
        with f.gui.sub_window("Renderer", 0.02, 0.02, 0.3, 0.2):
            f.gui.text(f.renderer._name())
            f.gui.text(f"Sampler: {f.renderer.params['sampler']}")
            spp = f.gui.slider_int("spp", f.renderer.params["samples_per_pixel"], 1, 10)
            if f.renderer.params["samples_per_pixel"] != spp:
                f.renderer.set_spp(spp)
                f.panel_update = True

            # Path tracer specific
            if isinstance(f.renderer, PathTracer):
                max_depth = f.gui.slider_int(
                    "depth", f.renderer.params["max_depth"], 1, 20
                )
                ambient_rate = f.gui.slider_float(
                    "ambient", f.renderer.params["ambient_rate"], 0.0, 1.0
                )
                direc_light_weight = f.gui.slider_float(
                    "direct light",
                    f.renderer.params["direct_light_weight"],
                    0.0,
                    10.0,
                )
                p_rr = f.gui.slider_float("p_rr", f.renderer.params["p_rr"], 0.0, 1.0)

                if f.renderer.params["max_depth"] != max_depth:
                    f.renderer.set_max_depth(max_depth)
                    f.panel_update = True
                if f.renderer.params["ambient_rate"] != ambient_rate:
                    f.renderer.set_ambient_rate(ambient_rate)
                    f.panel_update = True
                if f.renderer.params["direct_light_weight"] != direc_light_weight:
                    f.renderer.set_direct_light_weight(direc_light_weight)
                    f.panel_update = True
                if f.renderer.params["p_rr"] != p_rr:
                    f.renderer.set_prr(p_rr)
                    f.panel_update = True

            # ZBuffer specific
            if isinstance(f.renderer, ZBuffer):
                rate = f.gui.slider_float(
                    "alpha", f.renderer.params["alpha"], 0.1, 100.0
                )

                if f.renderer.params["alpha"] != rate:
                    f.renderer.set_alpha(rate)
                    f.panel_update = True

            # Blinn-Phong specific
            if isinstance(f.renderer, BlinnPhong):
                diffuse = f.gui.slider_float(
                    "diffuse", f.renderer.params["diffuse_rate"], 0.0, 1.0
                )
                ambient = f.gui.slider_float(
                    "ambient", f.renderer.params["ambient_rate"], 0.0, 1.0
                )
                click_enable_cosine = f.gui.button(
                    f"{'Cosine Enabled' if f.renderer.params['enable_cosine'] else 'Cosine Disabled'}",
                )
                if f.renderer.params["diffuse_rate"] != diffuse:
                    f.renderer.set_diffuse_rate(diffuse)
                    f.panel_update = True
                if f.renderer.params["ambient_rate"] != ambient:
                    f.renderer.set_ambient_rate(ambient)
                    f.panel_update = True
                if click_enable_cosine:
                    f.renderer.set_enable_cosine(not f.renderer.params["enable_cosine"])
                    f.panel_update = True

        # Camera
        with f.gui.sub_window("Camera", 0.02, 0.22, 0.3, 0.08):
            fov = f.gui.slider_float("fov", f.camera.params["fov"], 30.0, 150.0)
            if f.camera.params["fov"] != fov:
                f.camera.set_fov(fov)
                f.panel_update = True

        # Post Process
        for index, core in enumerate(f.post_processors):
            with f.gui.sub_window(
                f"Post Processor Core_{index}", 0.02, 0.3 + 0.12 * index, 0.3, 0.12
            ):
                f.gui.text(core._name())
                click_enable = f.gui.button(
                    f"{'Enabled' if core.params['enabled'] else 'Disabled'}",
                )
                if click_enable:
                    core.toggle()
                if isinstance(core, GaussianBlur):
                    radius = f.gui.slider_int("radius", core.params["radius"], 0, 10)
                    weight = f.gui.slider_float(
                        "weight", core.params["weight"], 0.0, 1.0
                    )

                    sigma = f.gui.slider_float("sigma", core.params["sigma"], 0.0, 5.0)

                    if core.params["radius"] != radius:
                        core.set_radius(radius)
                        f.panel_update = True
                    if core.params["weight"] != weight:
                        core.set_weight(weight)
                        f.panel_update = True
                    if core.params["sigma"] != sigma:
                        core.set_sigma(sigma)
                        f.panel_update = True

                if isinstance(core, BilateralFilter):
                    radius = f.gui.slider_int("radius", core.params["radius"], 0, 10)
                    weight = f.gui.slider_float(
                        "weight", core.params["weight"], 0.0, 1.0
                    )
                    sigma_d = f.gui.slider_float(
                        "sigma_d", core.params["sigma_d"], 0.0, 5.0
                    )
                    sigma_r = f.gui.slider_float(
                        "sigma_r", core.params["sigma_r"], 0.0, 5.0
                    )
                    if core.params["radius"] != radius:
                        core.set_radius(radius)
                        f.panel_update = True
                    if core.params["weight"] != weight:
                        core.set_weight(weight)
                        f.panel_update = True
                    if core.params["sigma_d"] != sigma_d:
                        core.set_sigma_d(sigma_d)
                        f.panel_update = True
                    if core.params["sigma_r"] != sigma_r:
                        core.set_sigma_r(sigma_r)
                        f.panel_update = True

                if isinstance(core, JointBilateralFilter):
                    sigma_z = f.gui.slider_float(
                        "sigma_z", core.params["sigma_z"], 0.0, 5.0
                    )
                    sigma_p = f.gui.slider_float(
                        "sigma_p", core.params["sigma_p"], 0.0, 5.0
                    )
                    sigma_n = f.gui.slider_float(
                        "sigma_n", core.params["sigma_n"], 0.0, 5.0
                    )
                    sigma_a = f.gui.slider_float(
                        "sigma_a", core.params["sigma_a"], 0.0, 5.0
                    )

                    if core.params["sigma_z"] != sigma_z:
                        core.set_sigma_z(sigma_z)
                        f.panel_update = True
                    if core.params["sigma_p"] != sigma_p:
                        core.set_sigma_p(sigma_p)
                        f.panel_update = True
                    if core.params["sigma_n"] != sigma_n:
                        core.set_sigma_n(sigma_n)
                        f.panel_update = True
                    if core.params["sigma_a"] != sigma_a:
                        core.set_sigma_a(sigma_a)
                        f.panel_update = True

                if isinstance(core, ToneMapping):
                    exposure = f.gui.slider_float(
                        "exposure", core.params["exposure"], 0.1, 10.0
                    )
                    if core.params["exposure"] != exposure:
                        core.set_exposure(exposure)
                        f.panel_update = True

        # Control Panel
        with f.gui.sub_window("Control Panel", 0.02, 0.85, 0.2, 0.12):
            max_fps = f.gui.slider_int("Target FPS", f.fps[None], 10, 90)
            panel_update = f.gui.button("Refresh")

            if panel_update:
                f.panel_update = True

            if f.fps[None] != max_fps:
                f.fps[None] = max_fps
                f._set_fps(max_fps)
                f.panel_update = True
