from abc import ABC, abstractmethod
from typing import Dict

import taichi as ti


@ti.data_oriented
class ProcessorCore(ABC):
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = ti.field(dtype=ti.i32, shape=())

        self.enabled[None] = 1 if enabled else 0

        self.params: Dict = {}
        self.update()

    def _name(self) -> str:
        return self.__class__.__name__

    def set_buffers(self, buffers) -> None:
        self.res = (buffers.shape[0], buffers.shape[1])

        self.buffers = buffers
        self.dst = ti.Vector.field(3, dtype=ti.f32, shape=buffers.shape)

    def toggle(self) -> None:
        self.enabled[None] = 0 if self.params["enabled"] else 1
        self.update()

    def update(self) -> None:
        self.params["enabled"] = self.enabled[None]

    def get_params(self) -> Dict:
        return self.params

    @abstractmethod
    @ti.kernel
    def process(self): ...
