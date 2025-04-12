import time
from typing import Dict

import numpy as np
import taichi as ti
from taichi.math import vec3
from taichi.ui.utils import euler_to_vec, vec_to_euler

from .objects import Ray


@ti.data_oriented
class Camera:
    def __init__(
        self,
        fov=60,
        aspect_ratio=1.0,
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

    def track(
        self,
        window: ti.ui.Window,
        movement_speed: float = 0.03,
        yaw_speed: float = 3.0,
        pitch_speed: float = 3.0,
    ) -> None:
        front = (self.lookat[None] - self.lookfrom[None]).normalized()
        position_change = vec3(0.0)
        left = self.vup[None].cross(front)

        if self.last_time is None:
            self.last_time = time.perf_counter_ns()
        time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
        self.last_time = time.perf_counter_ns()

        movement_speed *= time_elapsed * 60.0
        if window.is_pressed("w"):
            position_change += front * movement_speed
        if window.is_pressed("s"):
            position_change -= front * movement_speed
        if window.is_pressed("a"):
            position_change += left * movement_speed
        if window.is_pressed("d"):
            position_change -= left * movement_speed
        if window.is_pressed(ti.ui.SPACE):
            position_change += self.up[None] * movement_speed
        if window.is_pressed(ti.ui.SHIFT):
            position_change -= self.up[None] * movement_speed

        self.set_lookfrom(*(position_change + self.lookfrom[None]))

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()

        if window.is_pressed(ti.ui.LMB):
            if (self.last_mouse_x is None) or (self.last_mouse_y is None):
                self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
            dx = curr_mouse_x - self.last_mouse_x
            dy = curr_mouse_y - self.last_mouse_y

            yaw, pitch = vec_to_euler(front)

            yaw -= dx * yaw_speed * time_elapsed * 60.0
            pitch += dy * pitch_speed * time_elapsed * 60.0

            pitch_limit = np.pi / 2 * 0.99
            if pitch > pitch_limit:
                pitch = pitch_limit
            elif pitch < -pitch_limit:
                pitch = -pitch_limit

            front = euler_to_vec(yaw, pitch)

        self.set_lookat(*(self.lookfrom[None] + front))
        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y

    def focus_on(
        self,
        window: ti.ui.Window,
        yaw_speed: float = 3.0,
        pitch_speed: float = 3.0,
    ) -> None:
        front = (self.lookat[None] - self.lookfrom[None]).normalized()
        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        if self.last_time is None:
            self.last_time = time.perf_counter_ns()
        time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
        self.last_time = time.perf_counter_ns()
        if window.is_pressed(ti.ui.RMB):
            if (self.last_mouse_x is None) or (self.last_mouse_y is None):
                self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y

            dx = curr_mouse_x - self.last_mouse_x
            dy = curr_mouse_y - self.last_mouse_y

            yaw, pitch = vec_to_euler(front)

            yaw -= dx * yaw_speed * time_elapsed * 60.0
            pitch += dy * pitch_speed * time_elapsed * 60.0

            pitch_limit = np.pi / 2 * 0.99
            if pitch > pitch_limit:
                pitch = pitch_limit
            elif pitch < -pitch_limit:
                pitch = -pitch_limit

            front = euler_to_vec(yaw, pitch)

        self.set_lookfrom(*(self.lookat[None] - front * self.distance))
        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
        if self.distance <= 0.0:
            self.distance = 1e-3
