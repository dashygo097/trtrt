import datetime
import os
import sys
from typing import List

import taichi as ti


class InputTracer:
    def __init__(self, window: ti.ui.Window, pixels):
        self.window = window
        self.pixels = pixels
        self.holdkeys: List = [
            "w",
            "a",
            "s",
            "d",
            ti.ui.SHIFT,
            ti.ui.SPACE,
            ti.ui.RMB,
            ti.ui.LMB,
        ]

        self.show_panel: bool = False

    def on_move(self) -> bool:
        is_moving = False
        for i in range(8):
            if self.window.is_pressed(self.holdkeys[i]):
                is_moving = True

        return is_moving

    def refresh(self) -> bool:
        if self.window.is_pressed("r"):
            return True

        return False

    def control_panel(self) -> None:
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.TAB:
                self.show_panel = not self.show_panel

    def is_showing_panel(self):
        return self.show_panel

    def should_clear(self) -> bool:
        return (self.on_move() & (not self.is_showing_panel())) | self.refresh()

    def keymap(self):
        if self.window.is_pressed("j"):
            current_time = datetime.datetime.now()
            dirpath = sys.path[0]
            os.makedirs(f"{dirpath}/screenshots", exist_ok=True)
            fname = f"{dirpath}/screenshots/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            ti.tools.image.imwrite(self.pixels.to_numpy(), fname)  # pyright: ignore
            print(f"[INFO] Screenshot has been saved to {fname}!")

        if self.window.is_pressed("q"):
            self.window.running = False
