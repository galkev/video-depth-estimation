import bpy
import random
import numpy as np

camera = bpy.data.objects["Camera"]

"""
filename = "/home/kevin/Documents/master-thesis/code/blender/scene_shuffle.py"
exec(compile(open(filename).read(), filename, 'exec'))
"""

def random_cam():
    for i in range(0, 50000+1, 25):
        #bpy.context.scene.frame_set(i)

        new_rot = [
            np.random.uniform(1/4*np.pi, 3/4*np.pi),
            0,
            np.random.uniform(0, 2*np.pi)
        ]

        camera.rotation_euler = new_rot

        camera.keyframe_insert(
            data_path='rotation_euler',
            #index=2,
            frame=i)


def set_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def random_mat(obj):
    mat = random.choice(bpy.data.materials)
    set_material(obj, mat)


def random_rigidbody(obj):
    rigid_body = obj.rigid_body
    rigid_body.mass = random.uniform(0.01, 5)
    rigid_body.friction = random.random()
    rigid_body.restitution = random.random()


def main():
    #select = bpy.context.selected_objects
    rigid_bodies = [x for x in bpy.data.objects if x.rigid_body is not None and x.rigid_body.type == "ACTIVE"]

    for obj in rigid_bodies:
        random_mat(obj)
        random_rigidbody(obj)

    random_cam()


if __name__ == '__main__':
    main()
