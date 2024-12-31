# Copyright (c) 2020 Matthew Earl
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

import random
from dataclasses import dataclass
from typing import Dict, Set, List, Optional

import bmesh
import bpy
import bpy_types
import numpy as np

from . import mdl, blendmat


@dataclass
class PointLight:
    obj: bpy_types.Object
    locations: List[tuple]
    radii: List[float]
    parent_visible: bool = True
    pose_visible: bool = True

    def add_visible_keyframe_impl(self, blender_frame: int):
        self.obj.hide_render = not self.parent_visible or not self.pose_visible
        self.obj.keyframe_insert('hide_render', frame=blender_frame)
        self.obj.hide_viewport = not self.parent_visible or not self.pose_visible
        self.obj.keyframe_insert('hide_viewport', frame=blender_frame)

    def add_visible_keyframe(self, visible: bool, blender_frame: int):
        self.parent_visible = visible
        self.add_visible_keyframe_impl(blender_frame)

    def set_keyframe(self, pose_num: int, frame: int, hide: bool):
        self.obj.location = self.locations[pose_num]
        self.obj.data.shadow_soft_size = self.radii[pose_num]
        self.obj.keyframe_insert('location', frame=frame)
        self.obj.data.keyframe_insert('shadow_soft_size', frame=frame)
        self.pose_visible = not hide
        self.add_visible_keyframe_impl(frame)

    def set_sample_as_light(self, sample_as_light: bool):
        self.obj.data.cycles.use_multiple_importance_sampling = sample_as_light

    def add_sample_as_light_keyframe(self, sample_as_light: bool, blender_frame: int):
        self.set_sample_as_light(sample_as_light)
        self.obj.data.cycles.keyframe_insert("use_multiple_importance_sampling", frame=blender_frame)

    def __hash__(self):
        return self.obj.__hash__()

    def __eq__(self, other):
        # kinda questionable
        return self.obj == other.obj


@dataclass(frozen=True)
class BlendMdlSubobject:
    obj: bpy_types.Object
    shape_keys: List[bpy.types.ShapeKey]
    point_light: Optional[PointLight]
    _submdl_cfg: dict

    def add_visible_keyframe(self, visible: bool, blender_frame: int):
        self.obj.hide_render = not visible
        self.obj.keyframe_insert('hide_render', frame=blender_frame)
        self.obj.hide_viewport = not visible
        self.obj.keyframe_insert('hide_viewport', frame=blender_frame)
        if self.point_light:
            self.point_light.add_visible_keyframe(visible, blender_frame)

    def _update_pose(self, last_time: float, time: float, current_pose_num: int, pose_num: int, fps: float):
        blender_frame = int(round(fps * time))
        if current_pose_num is not None:
            self.shape_keys[current_pose_num].value = 0
            self.shape_keys[current_pose_num].keyframe_insert('value', frame=blender_frame)
            last_blender_frame = int(round(fps * last_time))
            self.shape_keys[pose_num].value = 0
            self.shape_keys[pose_num].keyframe_insert('value', frame=last_blender_frame)

        self.shape_keys[pose_num].value = 1
        self.shape_keys[pose_num].keyframe_insert('value', frame=blender_frame)

        if self.point_light:
            hide_point_light = pose_num in self._submdl_cfg.get("point_light_hide_in_pose", [])
            self.point_light.set_keyframe(pose_num, blender_frame, hide_point_light)

    def done(self, fps):
        if self.obj.data.shape_keys.animation_data:
            for c in self.obj.data.shape_keys.animation_data.action.fcurves:
                for kfp in c.keyframe_points:
                    kfp.interpolation = self._submdl_cfg.get('anim_interpolation', 'LINEAR')
        if self.point_light and self.point_light.obj.animation_data:
            for c in self.point_light.obj.animation_data.action.fcurves:
                for kfp in c.keyframe_points:
                    kfp.interpolation = self._submdl_cfg.get('anim_interpolation', 'LINEAR')
        if self.point_light and self._submdl_cfg.get('flicker', 0.0) != 0.0:
            self.point_light.obj.data.keyframe_insert('energy', frame=1)
            fcurve = self.point_light.obj.data.animation_data.action.fcurves[-1]
            noise = fcurve.modifiers.new(type='NOISE')
            noise.scale = fps
            noise.strength = self.point_light.obj.data.energy * self._submdl_cfg['flicker']
            noise.depth = 2
            noise.offset = random.randint(-fps * 1000, fps * 1000)
            noise.phase = random.gauss(mu=0, sigma=1e3)

    def set_invisible_to_camera(self):
        self.obj.visible_camera = False


@dataclass
class BlendMdl:
    am: "AliasMdl"
    obj: bpy_types.Object
    sub_objs: List[BlendMdlSubobject]
    sample_as_light_mats: Set[blendmat.BlendMat]

    _initial_pose_num: int
    _group_frame_times: Optional[List[float]]
    _mdl_cfg: dict
    _current_pose_num: Optional[int] = None
    _last_time: Optional[float] = None

    def _update_pose(self, time: float, pose_num: int, fps: float):
        if self._current_pose_num is None or self._current_pose_num != pose_num:
            for sub_obj in self.sub_objs:
                sub_obj._update_pose(self._last_time, time, self._current_pose_num, pose_num, fps)

            self._current_pose_num = pose_num
            self._last_time = time

    def add_pose_keyframe(self, pose_num: int, time: float, fps: float):
        if self._mdl_cfg.get('no_anim', False):
            pass

        elif self._group_frame_times is not None:
            if pose_num != self._initial_pose_num:
                raise Exception("Changing pose of a model whose initial pose is a group frame "
                                "is unsupported")
        else:
            self._update_pose(time, pose_num, fps)

    def set_invisible_to_camera(self):
        for sub_obj in self.sub_objs:
            sub_obj.set_invisible_to_camera()

    def done(self, final_time: float, fps: float):
        if self._mdl_cfg.get('no_anim', False):
            return

        if self._group_frame_times is not None:
            loop_time = -random.random() * self._group_frame_times[-1]
            while loop_time < final_time:
                for pose_num, offset in enumerate([0] + self._group_frame_times[:-1]):
                    self._update_pose(loop_time + offset, pose_num, fps)
                loop_time += self._group_frame_times[-1]

        for sub_obj in self.sub_objs:
            sub_obj.done(fps)


def _set_uvs(mesh, am, tri_set):
    mesh.uv_layers.new()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv[0]

    for bm_face, tri_idx in zip(bm.faces, tri_set):
        tcs = am.get_tri_tcs(tri_idx)

        for bm_loop, (s, t) in zip(reversed(bm_face.loops), tcs):
            bm_loop[uv_layer].uv = s / am.header['skin_width'], t / am.header['skin_height']

    bm.to_mesh(mesh)


def _simplify_pydata(verts, tris):
    vert_map = []
    new_tris = []
    for tri in tris:
        new_tri = []
        for vert_idx in tri:
            if vert_idx not in vert_map:
                vert_map.append(vert_idx)
            new_tri.append(vert_map.index(vert_idx))
        new_tris.append(new_tri)

    return ([verts[old_vert_idx] for old_vert_idx in vert_map], [], new_tris), vert_map


def _get_tri_set_fullbright_frac(am, tri_set, skin_idx):
    skin_area = 0
    fullbright_area = 0
    for tri_idx in tri_set:
        mask, skin = am.get_tri_skin(tri_idx, skin_idx)
        skin_area += np.sum(mask)
        fullbright_area += np.sum(mask * (skin >= 224))

    return fullbright_area / skin_area


def _create_shape_key(obj, simple_frame, vert_map):
    shape_key = obj.shape_key_add(name=simple_frame.name)
    for old_vert_idx, shape_key_vert in zip(vert_map, shape_key.data):
        shape_key_vert.co = simple_frame.frame_verts[old_vert_idx]
    return shape_key

def _get_geometric_center(vertices):
    return [sum(v.co[i] for v in vertices) / len(vertices) for i in range(3)]

def _get_distance(v, u):
    return np.linalg.norm(np.array(v) - np.array(u))

def _get_average_distance_from_center(vertices):
    center = _get_geometric_center(vertices)
    return sum(_get_distance(v.co, center) / len(vertices) for v in vertices)

def make_point_light(obj, subobj, mesh, shape_keys, scale, subobj_cfg):
    light = bpy.data.lights.new(name=subobj.name + '_pointlight', type='POINT')
    light.energy = subobj_cfg['strength'] * _get_average_distance_from_center(mesh.vertices) * scale
    light.color = subobj_cfg['tint'][:3]
    light_object = bpy.data.objects.new(name=subobj.name + '_pointlight', object_data=light)
    light_object.parent = obj
    subobj.visible_shadow = False
    center = _get_geometric_center(mesh.vertices)
    light_object.location = center
    bpy.context.scene.collection.objects.link(light_object)
    return PointLight(
        light_object,
        [_get_geometric_center(shape_key.data) for shape_key in shape_keys],
        [_get_average_distance_from_center(shape_key.data) * scale for shape_key in shape_keys],
    )


def add_model(am, pal, mdl_name, obj_name, skin_num, mdl_cfg, initial_pose_num, do_materials,
              known_materials: Dict[str, blendmat.BlendMat], scale: float):
    pal = np.concatenate([pal, np.ones(256)[:, None]], axis=1)

    # If the initial pose is a group frame, just load frames from that group.
    if am.frames[initial_pose_num].frame_type == mdl.FrameType.GROUP:
        group_frame = am.frames[initial_pose_num]
        timescale = mdl_cfg.get('timescale', 1)
        group_times = [t / timescale for t in group_frame.times]
    else:
        group_frame = None
        group_times = None
        for frame in am.frames:
            if frame.frame_type != mdl.FrameType.SINGLE:
                raise Exception(f"Frame type {frame.frame_type} not supported for non-static models")

    # Set up things specific to each tri-set
    sample_as_light_mats: Set[blendmat.BlendMat] = set()
    obj = bpy.data.objects.new(obj_name, None)
    sub_objs = []
    bpy.context.scene.collection.objects.link(obj)
    for tri_set_idx, tri_set in enumerate(am.disjoint_tri_sets):
        subobj_cfg = mdl_cfg
        if f"{mdl_name}_triset{tri_set_idx}" in mdl_cfg:
            subobj_cfg = dict(mdl_cfg)
            subobj_cfg.update(mdl_cfg.get(f"{mdl_name}_triset{tri_set_idx}", dict()))

        # Create the mesh and object
        subobj_name = f"{obj_name}_triset{tri_set_idx}"
        mesh = bpy.data.meshes.new(subobj_name)
        if am.frames[0].frame_type == mdl.FrameType.SINGLE:
            initial_verts = am.frames[0].frame.frame_verts
        else:
            initial_verts = am.frames[initial_pose_num].frames[0].frame_verts
        pydata, vert_map = _simplify_pydata([list(v) for v in initial_verts],
                                            [list(reversed(am.tris[t])) for t in tri_set])
        mesh.from_pydata(*pydata)
        if subobj_cfg['shade_smooth']:
            mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
        subobj = bpy.data.objects.new(subobj_name, mesh)
        subobj.parent = obj
        bpy.context.scene.collection.objects.link(subobj)

        # Create shape keys, used for animation.
        if group_frame is None:
            shape_keys = [
                _create_shape_key(subobj, frame.frame, vert_map) for frame in am.frames
            ]
        else:
            shape_keys = [
                _create_shape_key(subobj, simple_frame, vert_map)
                for simple_frame in group_frame.frames
            ]

        point_light = None
        if do_materials:
            # Set up material
            sample_as_light = subobj_cfg['sample_as_light']
            mat_name = f"{mdl_name}_skin{skin_num}"

            if subobj_cfg.get('point_light', False):
                point_light = make_point_light(obj, subobj, mesh, shape_keys, scale, subobj_cfg)
                point_light.set_sample_as_light(sample_as_light)
                if sample_as_light:
                    sample_as_light_mats.add(point_light)
                sample_as_light = False

            if sample_as_light:
                mat_name = f"{mat_name}_{obj_name}_triset{tri_set_idx}_fullbright"
            elif subobj_cfg != mdl_cfg:
                mat_name = f"{mdl_name}_triset{tri_set_idx}_skin{skin_num}"

            if mat_name not in known_materials:
                array_im, fullbright_array_im, _ = blendmat.array_ims_from_indices(
                    pal,
                    am.skins[skin_num],
                    force_fullbright=subobj_cfg['force_fullbright']
                )
                im = blendmat.im_from_array(mat_name, array_im)
                if subobj_cfg.get('point_light', False):
                    fullbright_im = blendmat.im_from_array(f"{mat_name}_fullbright", fullbright_array_im)
                    bm = blendmat.setup_fullbright_underlay_material(
                        blendmat.BlendMatImages.from_single_pair(im, fullbright_im),
                        mat_name,
                        subobj_cfg,
                        warp=False
                    )
                elif fullbright_array_im is not None:
                    fullbright_im = blendmat.im_from_array(f"{mat_name}_fullbright", fullbright_array_im)
                    bm = blendmat.setup_fullbright_material(
                        blendmat.BlendMatImages.from_single_pair(im, fullbright_im),
                        mat_name,
                        subobj_cfg,
                        warp=False
                    )
                else:
                    bm = blendmat.setup_diffuse_material(
                        blendmat.BlendMatImages.from_single_diffuse(im),
                        mat_name,
                        subobj_cfg,
                        warp=False
                    )
                bm.set_sample_as_light(sample_as_light)

                if sample_as_light:
                    sample_as_light_mats.add(bm)

                known_materials[mat_name] = bm
            bm = known_materials[mat_name]

            # Apply the material
            mesh.materials.append(bm.mat)
            _set_uvs(mesh, am, tri_set)

        sub_objs.append(BlendMdlSubobject(subobj, shape_keys, point_light, subobj_cfg))

    return BlendMdl(am, obj, sub_objs, sample_as_light_mats,
                    initial_pose_num, group_times, mdl_cfg)

