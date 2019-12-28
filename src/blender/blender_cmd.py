import bpy
import numpy as np
import os
import sys
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(bpy.data.filepath), "../code/blender/"))

import scene_gen
importlib.reload(scene_gen)
from scene_gen import ObjectPool

camera = bpy.data.objects["Camera"]
scene = bpy.context.scene

"""
filename = "/home/kevin/Documents/master-thesis/code/blender/blender_cmd.py"
exec(compile(open(filename).read(), filename, 'exec'))
"""

mt_root = \
        "/storage/slurm/galim/Documents/master-thesis" \
        if os.path.isdir("/storage/slurm") else "/home/kevin/Documents/master-thesis"


def bake_save(end=5000):
    print("Start bake")
    scene.rigidbody_world.point_cache.frame_start = 0
    scene.rigidbody_world.point_cache.frame_end = end

    override = {'scene': bpy.context.scene,
                'point_cache': bpy.context.scene.rigidbody_world.point_cache}
    # bake to current frame
    bpy.ops.ptcache.free_bake_all(override)
    bpy.ops.ptcache.bake_all(override, bake=True)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

    print("File saved")


def import_scene_pool_obj():
    for mode in ["train", "test"]:
        ObjectPool(os.path.join(mt_root, "scene_pool"), mode, test_run=False, load_all="disk", clear_obj=False)

    # scene_path = os.path.splitext(bpy.data.filepath)
    out_scene = os.path.join(mt_root, "scenes", "template_scene_import.blend")
    print(out_scene)

    bpy.ops.wm.save_as_mainfile(filepath=out_scene)


if __name__ == "__main__":
    import_scene_pool_obj()
    #bake_save()
    #random_cam()
