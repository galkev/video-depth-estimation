import os
#import bpy
import argparse


def get_cmd(scene, script, show_blender, script_args):
    blender = "/usr/stud/galim/blender-2.79b-linux-glibc219-x86_64/blender"

    if not os.path.isfile(blender):
        blender = "blender"

    # scene = "../../../scenes/casa2_auto.blend"

    return "{} {} {} --python {} -- {}".format(
        blender,
        scene,
        "--background" if not show_blender else "",
        script,
        " ".join(script_args)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--script", default="../blender/render.py")
    parser.add_argument("--scene", default="../../../scenes/template_scene.blend")
    parser.add_argument("--show_blender", action="store_true")

    args, script_args = parser.parse_known_args()

    cmd = get_cmd(args.scene, args.script, args.show_blender, script_args)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
