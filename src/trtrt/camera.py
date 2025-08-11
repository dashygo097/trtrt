import time
from typing import Dict

import numpy as np
import taichi as ti
from taichi.math import vec3

from .objects import Ray


@ti.func
def euler_to_vec(yaw, pitch):
    v = vec3(0.0)
    v[0] = ti.sin(yaw) * ti.cos(pitch)
    v[1] = ti.sin(pitch)
    v[2] = ti.cos(yaw) * ti.cos(pitch)
    return v


@ti.func
def vec_to_euler(v: vec3):
    v = v.normalized()
    pitch = ti.asin(v[1])

    cos_pitch = ti.sqrt(1 - v[1] * v[1])

    sin_yaw = v[0] / cos_pitch
    cos_yaw = v[2] / cos_pitch

    yaw = ti.atan2(sin_yaw, cos_yaw)

    return yaw, pitch


@ti.data_oriented
class Camera:
    def __init__(
        self,
        fov: int = 60,
        aspect_ratio: float = 1.0,
    ) -> None:
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.aspect_ratio = ti.field(dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.fov[None] = fov
        self.aspect_ratio[None] = aspect_ratio
        self.vup.from_numpy(np.array([0, 1, 0], dtype=np.float32))

        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.front = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.right = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.half_height = ti.field(dtype=ti.f32, shape=())
        self.half_width = ti.field(dtype=ti.f32, shape=())

        self.view = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.proj = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.prev_view_proj = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
        self.curr_view_proj = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())

        self.last_time = None
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.distance = 5.0

        self.params: Dict = {}
        self.params["fov"] = fov

    def set_fov(self, fov: float) -> None:
        self.fov[None] = fov
        self.params["fov"] = fov

    def set_lookfrom(self, x, y, z) -> None:
        self.lookfrom[None] = ti.Vector([x, y, z])

    def set_lookat(self, x, y, z) -> None:
        self.lookat[None] = ti.Vector([x, y, z])

    def set_distance(self, distance: float) -> None:
        self.distance = distance

    @ti.func
    def update(self) -> None:
        theta = self.fov[None] * np.pi / 180
        self.half_height[None] = ti.tan(theta / 2)
        self.half_width[None] = self.aspect_ratio[None] * self.half_height[None]
        self.front[None] = (self.lookfrom[None] - self.lookat[None]).normalized()
        self.right[None] = self.vup[None].cross(self.front[None]).normalized()
        self.up[None] = self.front[None].cross(self.right[None])
        self.horizontal[None] = 2 * self.half_width[None] * self.right[None]
        self.vertical[None] = 2 * self.half_height[None] * self.up[None]
        self.lower_left_corner[None] = (
            self.lookfrom[None]
            - self.half_width[None] * self.right[None]
            - self.half_height[None] * self.up[None]
            - self.front[None]
        )
        self.update_view()
        self.update_proj()
        self.prev_view_proj[None] = self.curr_view_proj[None]
        self.curr_view_proj[None] = self.proj[None] @ self.view[None]

    @ti.func
    def update_view(self):
        self.view[None][0, 0], self.view[None][0, 1], self.view[None][0, 2] = (
            self.right[None][0],
            self.right[None][1],
            self.right[None][2],
        )
        self.view[None][1, 0], self.view[None][1, 1], self.view[None][1, 2] = (
            self.up[None][0],
            self.up[None][1],
            self.up[None][2],
        )
        self.view[None][2, 0], self.view[None][2, 1], self.view[None][2, 2] = (
            self.front[None][0],
            self.front[None][1],
            self.front[None][2],
        )
        self.view[None][0, 3] = -self.right[None].dot(self.lookfrom[None])
        self.view[None][1, 3] = -self.up[None].dot(self.lookfrom[None])
        self.view[None][2, 3] = -self.front[None].dot(self.lookfrom[None])
        self.view[None][3, 3] = 1.0

    @ti.func
    def update_proj(self):
        self.proj[None][0, 0] = 1.0 / (self.aspect_ratio[None] * self.half_width[None])
        self.proj[None][1, 1] = 1.0 / self.half_width[None]
        self.proj[None][2, 2] = -1.0
        self.proj[None][2, 3] = -1.0
        self.proj[None][3, 2] = -1.0

    @ti.kernel
    def update_track(
        self,
        lookfrom: ti.template(),
        lookat: ti.template(),
        vup: ti.template(),
        up: ti.template(),
        movement: ti.f32,
        yaw_delta: ti.f32,
        pitch_delta: ti.f32,
        w_pressed: ti.i32,
        s_pressed: ti.i32,
        a_pressed: ti.i32,
        d_pressed: ti.i32,
        space_pressed: ti.i32,
        shift_pressed: ti.i32,
        mouse_moved: ti.i32,
    ):
        front = (lookat[None] - lookfrom[None]).normalized()
        left = vup[None].cross(front)
        position_change = ti.math.vec3(0.0)

        if w_pressed:
            position_change += front * movement
        if s_pressed:
            position_change -= front * movement
        if a_pressed:
            position_change += left * movement
        if d_pressed:
            position_change -= left * movement
        if space_pressed:
            position_change += up[None] * movement
        if shift_pressed:
            position_change -= up[None] * movement

        lookfrom[None] += position_change
        lookat[None] += position_change

        if mouse_moved:
            yaw, pitch = vec_to_euler(front)
            yaw += yaw_delta
            pitch += pitch_delta

            pitch = ti.max(-1.5533, ti.min(1.5533, pitch))

            new_front = euler_to_vec(yaw, pitch)
            lookat[None] = lookfrom[None] + new_front

    def track(
        self,
        window: ti.ui.Window,
        movement_speed: float = 3.0,
        yaw_speed: float = 240.0,
        pitch_speed: float = 240.0,
    ) -> None:
        curr_time = time.perf_counter_ns()
        if self.last_time is None:
            self.last_time = curr_time
        time_elapsed = (curr_time - self.last_time) * 1e-9
        self.last_time = curr_time

        movement = movement_speed * time_elapsed

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        yaw_delta = pitch_delta = 0.0
        mouse_moved = 0

        if window.is_pressed(ti.ui.LMB):
            if self.last_mouse_x is not None and self.last_mouse_y is not None:
                dx = curr_mouse_x - self.last_mouse_x
                dy = curr_mouse_y - self.last_mouse_y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    yaw_delta = -dx * yaw_speed * time_elapsed
                    pitch_delta = dy * pitch_speed * time_elapsed
                    mouse_moved = 1
            self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
        else:
            self.last_mouse_x = self.last_mouse_y = None

        self.update_track(
            self.lookfrom,
            self.lookat,
            self.vup,
            self.up,
            movement,
            yaw_delta,
            pitch_delta,
            int(window.is_pressed("w")),
            int(window.is_pressed("s")),
            int(window.is_pressed("a")),
            int(window.is_pressed("d")),
            int(window.is_pressed(ti.ui.SPACE)),
            int(window.is_pressed(ti.ui.SHIFT)),
            mouse_moved,
        )

    @ti.func
    def get_ray(self, u, v) -> Ray:
        self.update()
        return Ray(
            self.lookfrom[None],
            self.lower_left_corner[None]
            + u * self.horizontal[None]
            + v * self.vertical[None]
            - self.lookfrom[None],
        )
