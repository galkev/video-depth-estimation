import bpy
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(bpy.data.filepath), "../code/blender/"))

import scene_config
importlib.reload(scene_config)
from scene_config import SceneConfig

"""
filename = "/home/kevin/Documents/master-thesis/code/blender/load_render_setup.py"
exec(compile(open(filename).read(), filename, 'exec'))
"""


if __name__ == "__main__":
    scene_config = SceneConfig()
    scene_config.load(0)
