import bpy
import random
import numpy as np
#import mathutils
import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(bpy.data.filepath), "../code/blender/"))

import scene_config
importlib.reload(scene_config)
from scene_config import SceneConfig

"""
filename = "/home/kevin/Documents/master-thesis/code/blender/random_scene.py"
exec(compile(open(filename).read(), filename, 'exec'))
"""

scene = bpy.context.scene
node_tree = scene.node_tree
camera = bpy.data.objects["Camera"]


def shift_vec(vec, strength):
    return list(np.array(vec) + strength * np.random.randn(3))


def shift_transform(obj, trans_strength=0.1, rot_strength=0.05):
    return {
        "location": shift_vec(obj.location, trans_strength),
        "rotation_euler": shift_vec(obj.rotation_euler, rot_strength)
    }


def random_transform(lower_pos, upper_pos):
    return {
        "location": [np.random.uniform(lower_pos[i], upper_pos[i]) for i in range(3)],
        "rotation_euler": list(np.random.uniform(0, 2*np.pi, 3))
    }


def random_scene():
    #random.seed(seed)
    #np.random.seed(seed)

    config = {}

    #print(camera.rotation_euler)
    #config["Camera"] = shift_transform(camera)
    #config["Camera"]["rotation_euler"][1] = 0

    rigid_bodies_names = [k for k, v in bpy.data.objects.items() if v.rigid_body is not None]

    for rigid_body_name in rigid_bodies_names:
        config[rigid_body_name] = {"rigid_body": {
                "mass": random.uniform(0.01, 5),
                "friction": random.random(),
                "restitution": random.random()
            }
        }

    """
    #field = bpy.data.objects["Field"]
    config["Field"] = {
        **random_transform(lower_pos=(-3, -3, 0), upper_pos=(3, 3, 3)),
        "field": {
            "size": random.uniform(0.0, 10),
            "strength": random.uniform(0.1, 10),
            "flow": random.uniform(0.0, 10),
            "noise": random.uniform(0, 10),
            "seed": random.randint(0, 128),
            "shape": random.choice(["POINT", "PLANE"]),
            "type": random.choice(["FORCE", "WIND", "VORTEX", "MAGNET", "HARMONIC", "CHARGE", "LENNARDJ",
                                   "TEXTURE", "GUIDE", "BOID", "TURBULENCE", "DRAG", "SMOKE_FLOW"]),
            "use_global_coords": random.randint(0, 1)
        }
    }
    """

    cfg = SceneConfig()
    cfg.set(config)

    return cfg

bl_info = {
    "name": "Random Scene",
    "category": "Object",
}


class SceneConfigProp:
    scene_config = None


class RandomScene(bpy.types.Operator):
    bl_idname = "object.random_scene"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Randomize Scene"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):  # execute() is called when running the operator.
        global scene_config
        bpy.types.Scene.sc_prop.scene_config = random_scene()
        return {'FINISHED'}


class SaveScene(bpy.types.Operator):
    bl_idname = "object.save_config"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Save Config"  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):  # execute() is called when running the operator.
        bpy.types.Scene.sc_prop.scene_config.update()
        id = bpy.types.Scene.sc_prop.scene_config.save()
        self.report({'INFO'}, "Save Config " + str(id))
        return {'FINISHED'}


addon_keymaps = []


def register():
    bpy.types.Scene.sc_prop = SceneConfigProp()

    bpy.utils.register_class(RandomScene)
    bpy.utils.register_class(SaveScene)

    # handle the keymap
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
    km.keymap_items.new(RandomScene.bl_idname, 'F9', 'PRESS', ctrl=False, shift=False)
    km.keymap_items.new(SaveScene.bl_idname, 'F10', 'PRESS', ctrl=False, shift=False)
    addon_keymaps.append(km)


def unregister():
    bpy.utils.unregister_class(RandomScene)

    # handle the keymap
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    # clear the list
    del addon_keymaps[:]

    del bpy.types.Scene.sc_prop


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()


#def main():
    #print(get_curve_point(bpy.data.objects["CamPath0"], 0.5))
    #camera.location = get_curve_point(bpy.data.objects["CamPath0"], random.random())
    #random_scene()


#if __name__ == "__main__":
    #main()


"""
def get_curve_points_discrete(obj, res=12):
    # Assumes your active object is a Bezier Curve.
    curve = obj.data

    # Assumes your Bezier is composed of only one spline.
    spline = curve.splines[0]
    spline_points = list(spline.bezier_points)

    if spline.use_cyclic_u:
        spline_points.append(spline_points[0])

    segmentResults = []

    # Iterate the control points in the spline and interpolate them.
    for i in range(1, len(spline_points)):
        # You always need at least 2 points to interpolate between.  Get the first and
        # second points for this segment of the spline.
        firstPt = spline_points[i - 1]
        secondPt = spline_points[i]

        # Get all the points on the curve between these two items.  Uses the default of 12 for a "preview" resolution
        # on the curve.  Note the +1 because the "preview resolution" tells how many segments to use.  ie. 2 =&gt; 2 segments
        # or 3 points.  The "interpolate_bezier" functions takes the number of points it should generate.
        segmentResults.extend(
            mathutils.geometry.interpolate_bezier(firstPt.co, firstPt.handle_right, secondPt.handle_left, secondPt.co,
                                                  res + 1))

    #result = [np.array(r) for r in segmentResults]
    result = np.array(segmentResults)

    return result


# t in [0, 1]
def get_curve_point(obj, t):
    points = get_curve_points_discrete(obj)
    print(points)
    return (np.array(obj.matrix_world) @ np.array(
        [np.interp(t, np.linspace(0, 1, len(points[:, d])), points[:, d]) for d in range(3)] + [1]))[:3]


def set_cam_pos(i, t):
    pass
"""