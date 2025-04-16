from typing import Tuple

import taichi as ti
from taichi.math import vec2, vec3, vec4


@ti.dataclass
class HitInfo:
    is_hit: ti.u1
    time: ti.f32
    pos: vec3
    normal: vec3
    front: ti.u1
    tag: ti.u32
    id: ti.u32
    u: ti.f32
    v: ti.f32

    albedo: vec3
    metallic: ti.f32
    roughness: ti.f32
    emission: vec3


@ti.dataclass
class BVHHitInfo:
    is_hit: ti.u1
    tmin: ti.f32
    tmax: ti.f32
    obj_id: ti.i32


@ti.dataclass
class GBuffer:
    depth: ti.f32
    pos: vec3
    normal: vec3
    albedo: vec3


@ti.data_oriented
class VelocityBuffer:
    def __init__(self, res: Tuple[int, int], camera):
        self.camera = camera
        self.width = res[0]
        self.height = res[1]

        self.current_positions = ti.Vector.field(3, dtype=ti.f32, shape=res)
        self.previous_positions = ti.Vector.field(3, dtype=ti.f32, shape=res)
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=res)

    @ti.kernel
    def compute_velocity(self):
        for i, j in self.velocity:
            if (
                self.current_positions[i, j][2] > 0.0
                and self.previous_positions[i, j][2] > 0.0
            ):
                curr_pos = self.current_positions[i, j].xy * 0.5 + 0.5
                prev_pos = self.previous_positions[i, j].xy * 0.5 + 0.5

                self.velocity[i, j] = (curr_pos - prev_pos) * ti.Vector(
                    [self.width, self.height]
                )
            else:
                self.velocity[i, j] = vec2(0.0)

    @ti.func
    def project_world_position(self, world_pos: vec3, matrix):
        pos = vec4(world_pos[0], world_pos[1], world_pos[2], 1.0)
        ndc_pos = vec3(0.0)

        clip_pos = matrix @ pos

        if clip_pos[3] != 0.0:
            ndc_pos = clip_pos.xyz / clip_pos[3]

        return ndc_pos

    @ti.kernel
    def store_positions(self, g_buffer: ti.template()):
        for i, j in g_buffer:
            self.previous_positions[i, j] = self.current_positions[i, j]
            self.current_positions[i, j] = self.project_world_position(
                g_buffer.pos[i, j], self.camera.curr_view_proj[None]
            )

    @ti.kernel
    def render_velocity(self, canvas: ti.template()):
        for i, j in canvas:
            # NOTE: The scale factor matters
            scale_factor = 0.02

            r = ti.abs(self.velocity[i, j][0]) * scale_factor
            g = ti.abs(self.velocity[i, j][1]) * scale_factor

            canvas[i, j] = ti.Vector([r, g, 0.0])
