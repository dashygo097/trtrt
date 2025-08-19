from ..postprocess import (BilateralFilter, GaussianBlur, JointBilateralFilter,
                           ToneMapping)
from ..renderer import BlinnPhong, PathTracer, ZBuffer
from .builder import UIBuilder


@UIBuilder.register_renderer(PathTracer)
def path_tracer_ui(frontend):
    max_depth = frontend.gui.slider_int(
        "depth", frontend.renderer.params["max_depth"], 1, 20
    )
    ambient_rate = frontend.gui.slider_float(
        "ambient", frontend.renderer.params["ambient_rate"], 0.0, 1.0
    )
    direc_light_weight = frontend.gui.slider_float(
        "direct light", frontend.renderer.params["direct_light_weight"], 0.0, 10.0
    )
    p_rr = frontend.gui.slider_float("p_rr", frontend.renderer.params["p_rr"], 0.0, 1.0)
    if frontend.renderer.params["max_depth"] != max_depth:
        frontend.renderer.set_max_depth(max_depth)
        frontend.panel_update = True
    if frontend.renderer.params["ambient_rate"] != ambient_rate:
        frontend.renderer.set_ambient_rate(ambient_rate)
        frontend.panel_update = True
    if frontend.renderer.params["direct_light_weight"] != direc_light_weight:
        frontend.renderer.set_direct_light_weight(direc_light_weight)
        frontend.panel_update = True
    if frontend.renderer.params["p_rr"] != p_rr:
        frontend.renderer.set_prr(p_rr)
        frontend.panel_update = True


@UIBuilder.register_renderer(ZBuffer)
def zbuffer_ui(frontend):
    rate = frontend.gui.slider_float(
        "alpha", frontend.renderer.params["alpha"], 0.1, 100.0
    )
    if frontend.renderer.params["alpha"] != rate:
        frontend.renderer.set_alpha(rate)
        frontend.panel_update = True


@UIBuilder.register_renderer(BlinnPhong)
def blinn_phong_ui(frontend):
    diffuse = frontend.gui.slider_float(
        "diffuse", frontend.renderer.params["diffuse_rate"], 0.0, 1.0
    )
    ambient = frontend.gui.slider_float(
        "ambient", frontend.renderer.params["ambient_rate"], 0.0, 1.0
    )
    click_enable_cosine = frontend.gui.button(
        f"{'Cosine Enabled' if frontend.renderer.params['enable_cosine'] else 'Cosine Disabled'}",
    )
    if frontend.renderer.params["diffuse_rate"] != diffuse:
        frontend.renderer.set_diffuse_rate(diffuse)
        frontend.panel_update = True
    if frontend.renderer.params["ambient_rate"] != ambient:
        frontend.renderer.set_ambient_rate(ambient)
        frontend.panel_update = True
    if click_enable_cosine:
        frontend.renderer.set_enable_cosine(
            not frontend.renderer.params["enable_cosine"]
        )
        frontend.panel_update = True


@UIBuilder.register_post(GaussianBlur)
def gaussian_blur_ui(frontend, core):
    radius = frontend.gui.slider_int("radius", core.params["radius"], 0, 10)
    weight = frontend.gui.slider_float("weight", core.params["weight"], 0.0, 1.0)
    sigma = frontend.gui.slider_float("sigma", core.params["sigma"], 0.0, 5.0)
    if core.params["radius"] != radius:
        core.set_radius(radius)
        frontend.panel_update = True
    if core.params["weight"] != weight:
        core.set_weight(weight)
        frontend.panel_update = True
    if core.params["sigma"] != sigma:
        core.set_sigma(sigma)
        frontend.panel_update = True


@UIBuilder.register_post(BilateralFilter)
def bilateral_filter_ui(frontend, core):
    radius = frontend.gui.slider_int("radius", core.params["radius"], 0, 10)
    weight = frontend.gui.slider_float("weight", core.params["weight"], 0.0, 1.0)
    sigma_d = frontend.gui.slider_float("sigma_d", core.params["sigma_d"], 0.0, 5.0)
    sigma_r = frontend.gui.slider_float("sigma_r", core.params["sigma_r"], 0.0, 5.0)
    if core.params["radius"] != radius:
        core.set_radius(radius)
        frontend.panel_update = True
    if core.params["weight"] != weight:
        core.set_weight(weight)
        frontend.panel_update = True
    if core.params["sigma_d"] != sigma_d:
        core.set_sigma_d(sigma_d)
        frontend.panel_update = True
    if core.params["sigma_r"] != sigma_r:
        core.set_sigma_r(sigma_r)
        frontend.panel_update = True


@UIBuilder.register_post(JointBilateralFilter)
def joint_bilateral_filter_ui(frontend, core):
    sigma_z = frontend.gui.slider_float("sigma_z", core.params["sigma_z"], 0.0, 5.0)
    sigma_p = frontend.gui.slider_float("sigma_p", core.params["sigma_p"], 0.0, 5.0)
    sigma_n = frontend.gui.slider_float("sigma_n", core.params["sigma_n"], 0.0, 5.0)
    sigma_a = frontend.gui.slider_float("sigma_a", core.params["sigma_a"], 0.0, 5.0)
    if core.params["sigma_z"] != sigma_z:
        core.set_sigma_z(sigma_z)
        frontend.panel_update = True
    if core.params["sigma_p"] != sigma_p:
        core.set_sigma_p(sigma_p)
        frontend.panel_update = True
    if core.params["sigma_n"] != sigma_n:
        core.set_sigma_n(sigma_n)
        frontend.panel_update = True
    if core.params["sigma_a"] != sigma_a:
        core.set_sigma_a(sigma_a)
        frontend.panel_update = True


@UIBuilder.register_post(ToneMapping)
def tone_mapping_ui(frontend, core):
    exposure = frontend.gui.slider_float("exposure", core.params["exposure"], 0.1, 10.0)
    if core.params["exposure"] != exposure:
        core.set_exposure(exposure)
        frontend.panel_update = True
