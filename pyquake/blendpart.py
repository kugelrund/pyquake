# Copyright (c) 2021 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


__all__ = (
    'Particles',
)

import dataclasses
import random
from typing import Dict, List

import bpy
import bpy_types
import bmesh
import numpy as np

from . import blendmat


@dataclasses.dataclass
class _Ramp:
    color_indices: List[int]
    speed: float
    # index of first element in color_indices where an animation with this ramp
    # can start
    start_min: int
    # index of one past the last element in color_indices where an animation
    # with this ramp can start
    start_max: int


@dataclasses.dataclass
class _Lifetime:
    min: float
    max: float

    @classmethod
    def from_ramp(cls, ramp: _Ramp):
        return _Lifetime(
            min=(len(ramp.color_indices) - ramp.start_max) / ramp.speed,
            max=(len(ramp.color_indices) - ramp.start_min) / ramp.speed,
        )


_EXPLOSION_RAMP = _Ramp(
    color_indices=[0x6f, 0x6d, 0x6b, 0x69, 0x67, 0x65, 0x63, 0x61],
    speed=10,
    start_min=0,
    start_max=4,
)
_EXPLOSION_LIFETIME = _Lifetime.from_ramp(_EXPLOSION_RAMP)

_EXPLOSION2_RAMP = _Ramp(
    color_indices=[0x6f, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x68, 0x66],
    speed=15,
    start_min=0,
    start_max=4,
)
_EXPLOSION2_LIFETIME = _Lifetime.from_ramp(_EXPLOSION2_RAMP)

_TELEPORT_COLOR_INDICES = list(range(7, 15))
_TELEPORT_LIFETIME = _Lifetime(min=0.2, max=0.34)


class Particles:
    root: bpy_types.Object
    _teleport: bpy_types.Object
    _explosion: bpy_types.Object
    _explosion2: bpy_types.Object
    _random_velocity_texture: bpy_types.Texture

    def __init__(self, pal, fps, scale):
        self.root = _create_particle_root()
        self._fps = fps
        self._scale = scale

        self._explosion = _create_explosion_particle_object("explosion",
            colors=pal[_EXPLOSION_RAMP.color_indices],
            max_lifetime_frames=int(round(_EXPLOSION_LIFETIME.max * fps))
        )
        self._explosion2 = _create_explosion_particle_object("explosion2",
            colors=pal[_EXPLOSION2_RAMP.color_indices],
            max_lifetime_frames=int(round(_EXPLOSION2_LIFETIME.max * fps))
        )
        self._teleport = _create_teleport_particle_object(
            colors=pal[_TELEPORT_COLOR_INDICES]
        )
        self._explosion.parent = self.root
        self._explosion2.parent = self.root
        self._teleport.parent = self.root

        tex = bpy.data.textures.new('explosion_velocity', type='CLOUDS')
        tex.intensity = 0.8
        tex.contrast = 1.2
        tex.noise_scale = 0.1
        self._random_velocity_texture = tex

    def _create_common_particle_system(
            self, emitter, name, pos, count, start_time, end_time,
            lifetime: _Lifetime, instance_object, gravity_weight):
        emitter.modifiers.new(f"{name}_particle_system", type='PARTICLE_SYSTEM')
        assert len(emitter.particle_systems) == 1
        max_seed = 2147483647
        emitter.particle_systems[0].seed = random.randrange(max_seed)

        part = emitter.particle_systems[0].settings
        part.count = count
        part.frame_start = int(round(start_time * self._fps))
        part.frame_end = int(round(end_time * self._fps))
        part.lifetime = int(round(lifetime.max * self._fps))
        part.lifetime_random = (lifetime.max - lifetime.min) / lifetime.max
        part.normal_factor = 0
        part.emit_from = 'VOLUME'
        part.distribution = 'RAND'
        part.render_type = 'OBJECT'
        part.instance_object = instance_object
        part.instance_object.parent = self.root
        part.particle_size = 1
        part.effector_weights.gravity = gravity_weight
        part.timestep = 1 / self._fps

        emitter.show_instancer_for_render = False
        emitter.show_instancer_for_viewport = False
        emitter.show_bounds = True
        emitter.parent = self.root
        emitter.location = pos

        return emitter

    def _create_single_explosion(self, name, obj_name, pos, start_time,
                                 lifetime, instance_object):
        emitter = _create_cuboid((-16, -16, -16), (15, 15, 15), obj_name)
        self._create_common_particle_system(emitter, name, pos,
            count=512,
            start_time=start_time,
            end_time=start_time,
            lifetime=lifetime,
            instance_object=instance_object,
            gravity_weight=30.0,
        )
        part = emitter.particle_systems[0].settings
        # this extra factor is not based in vanilla source code, but for some
        # reason it makes it look much more similar than without it
        extra_factor = 3.5
        part.factor_random = 256 * self._scale * extra_factor
        part_texture = part.texture_slots.add()
        part_texture.blend_type = 'MULTIPLY'
        part_texture.texture = self._random_velocity_texture
        part_texture.texture_coords = 'ORCO'
        part_texture.use_map_time = False
        part_texture.use_map_velocity = True

        return emitter

    def create_explosion(self, start_time, obj_name, pos):
        emitter = self._create_single_explosion("explosion", obj_name, pos,
            start_time, lifetime=_EXPLOSION_LIFETIME,
            instance_object=self._explosion)

        emitter2 = self._create_single_explosion("explosion2", f"{obj_name}_2",
            pos, start_time, lifetime=_EXPLOSION2_LIFETIME,
            instance_object=self._explosion2)
        emitter2.particle_systems[0].settings.damping = 0.2

        return emitter, emitter2

    def create_teleport(self, start_time, obj_name, pos):
        emitter = _create_cuboid((-16, -16, -24), (15, 15, 31), obj_name)
        self._create_common_particle_system(emitter, "teleport", pos,
            count=896,
            start_time=start_time,
            end_time=start_time,
            lifetime=_TELEPORT_LIFETIME,
            instance_object=self._teleport,
            gravity_weight=0.0
        )
        part = emitter.particle_systems[0].settings
        part.normal_factor = -48 * self._scale
        part.object_align_factor[2] = 16 * self._scale
        part.factor_random = -32 * self._scale

        return emitter


def _create_icosphere(diameter, obj_name):
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=diameter / 2)
    bm.to_mesh(mesh)
    bm.free()

    return obj


def _create_cuboid(mins, maxs, obj_name):
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    mins = np.array(mins)
    maxs = np.array(maxs)
    size = maxs - mins
    centre = 0.5 * (maxs + mins)

    bm = bmesh.new()
    d = bmesh.ops.create_cube(bm, size=1)
    verts = d['verts']
    bmesh.ops.scale(bm, vec=size, verts=verts)
    bmesh.ops.translate(bm, vec=centre, verts=verts)
    bm.to_mesh(mesh)
    bm.free()

    return obj


def _create_particle_root():
    obj = bpy.data.objects.new('particle_root', None)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _create_explosion_particle_object(name, colors, max_lifetime_frames):
    obj = _create_icosphere(1, f'{name}_particle')
    obj.data.materials.append(blendmat.setup_explosion_particle_material(
        name, colors, max_lifetime_frames).mat)
    obj.hide_render = True

    return obj


def _create_teleport_particle_object(colors):
    obj = _create_icosphere(1, 'teleport_particle')
    obj.data.materials.append(blendmat.setup_teleport_particle_material(
        'teleport', colors).mat)
    obj.hide_render = True

    return obj

