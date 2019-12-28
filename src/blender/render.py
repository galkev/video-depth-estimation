import sys
import os
import bpy
import importlib
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from blender import render_setup
from blender import scene_config
from blender import scene_gen
from blender.scene_gen import SceneGen

importlib.reload(render_setup)
importlib.reload(scene_config)
importlib.reload(scene_gen)

def main():
    print("yes")

    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles_depth", action="store_true")
    parser.add_argument("--gen_mode", required=True)  # train or test
    parser.add_argument("--obj_from_disk", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--setup_test", action="store_true")
    parser.add_argument("--job_name", default=None)
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        config = json.load(f)

    scene_generate = SceneGen(mode=args.gen_mode, mt_root=render_setup.RenderSetup.mt_root, config=config)
    scene_generate.scene_init(load_all_obj="disk" if args.obj_from_disk else "scene")

    render_dir = os.path.splitext(os.path.basename(args.config))[0]
    render_dir = os.path.join(render_dir, args.gen_mode, "seq{index}")

    for i in range(999999999 if not args.setup_test else 1):
        rendering = render_setup.RenderSetupDiningRoom(
            qual_mode="medium512",
            sub_dir=render_dir,
            use_blender_render_depth=not args.cycles_depth,
            config=config,
            scene_generate=scene_generate,
            job_name=args.job_name
        )

        rendering.setup_test = args.setup_test

        rendering.start()

        scene_generate.scene_shuffle()

        rendering.render_seq()


if __name__ == "__main__":
    main()
