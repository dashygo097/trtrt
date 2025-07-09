from typing import Dict

import taichi as ti

from .post_processor import ProcessorCore


@ti.data_oriented
class ToneMapping(ProcessorCore):
    def __init__(self, enabled: bool = False, exposure: float = 1.0) -> None:
        self.exposure = ti.field(dtype=ti.f32, shape=())

        self.exposure[None] = exposure

        self.params: Dict = {}
        super().__init__(enabled)

    def _name(self) -> str:
        return "Tone Mapping"

    def set_exposure(self, exposure: float) -> None:
        self.exposure[None] = exposure
        self.update()

    def update(self):
        super().update()
        # Update params
        self.params["exposure"] = self.exposure[None]

    @ti.kernel
    def process(self):
        for i, j in self.buffers:
            self.buffers[i, j] = (
                self.buffers[i, j]
                * self.exposure[None]
                / (self.buffers[i, j] * self.exposure[None] + 1.0)
            )


@ti.data_oriented
class HDR(ToneMapping):
    def __init__(self, enabled: bool = False, exposure: float = 1.0) -> None:
        super().__init__(enabled, exposure)

    def _name(self) -> str:
        return "HDR"

    @ti.kernel
    def process(self):
        for i, j in self.buffers:
            self.buffers[i, j] = 1 - ti.exp(-self.buffers[i, j] * self.exposure[None])
