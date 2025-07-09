import taichi as ti

from .const import ObjectShape


@ti.data_oriented
class Abstraction:
    def __init__(self, obj):
        self.entity = obj
        self.tag = obj.tag
        self.shape = self.map_shape(obj)

    def map_shape(self, obj):
        from ..objects import Sphere, Triangle

        if isinstance(obj, Sphere):
            return ObjectShape.SPHERE
        elif isinstance(obj, Triangle):
            return ObjectShape.TRIANGLE
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")
