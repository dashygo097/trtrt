from enum import Enum

import taichi as ti
from taichi.math import vec3

EPSILON = 1e-6
TMIN = 1e-3
TMAX = 1e8
NEAR_Z = 1e-1
FAR_Z = 5e2


@ti.data_oriented
class ObjectTag:
    PBR = 0
    GLASS = 1


@ti.data_oriented
class ObjectShape(Enum):
    TRIANGLE = 0
    SPHERE = 1
    CUBE = 2


@ti.data_oriented
class PBRPreset(Enum):
    MATTE = {
        "metallic": 0.0,
        "roughness": 1.0,
        "emission": vec3(0.0),
    }
    CONCRETE = {
        "metallic": 0.0,
        "roughness": 0.8,
        "emission": vec3(0.0),
    }
    MIRROR = {
        "metallic": 1.0,
        "roughness": 0.0,
        "emission": vec3(0.0),
    }
    SILVER = {
        "metallic": 1.0,
        "roughness": 0.1,
        "emission": vec3(0.0),
    }
