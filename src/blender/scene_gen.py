import bpy
#import glob
import random
import numpy as np
import os
import mathutils
import json
import time
import sys


sys.path.insert(0, os.path.expanduser("~/Documents/master-thesis/code/mt-project"))

from blender import lens_tools

"""
filename = "/home/kevin/Documents/master-thesis/code/mt-project/blender/scene_gen.py"; exec(compile(open(filename).read(), filename, 'exec'))
"""


def _set_obj_visible(obj, visible):
    obj.hide = not visible
    obj.hide_render = not visible


class ObjectPool(object):
    obj_name = "gen_obj"

    def __init__(self, scene_pool_path, mode, config, load_all=None, clear_obj=True):
        self.config = config

        self.scene_pool_path = scene_pool_path
        self.mode = mode
        self.obj_paths = self.get_file_paths()
        self.objects = [None] * len(self.obj_paths)

        self.load_all = load_all

        self.active_objects = None

        if clear_obj:
            for obj in bpy.data.objects:
                if obj.name.startswith(ObjectPool.obj_name):
                    bpy.data.objects.remove(obj)

        if self.load_all == "scene":
            self._load_all_from_scene()
        else:
            if self.load_all == "disk":
                t1 = time.time()
                self._load_all_from_disk()
                print("Loaded", len(self.objects))
                print("Loaded obj", time.time() - t1, "sec")

    def get_file_paths(self):
        with open(os.path.join(self.scene_pool_path, "split.json"), "r") as f:
            data = json.load(f)[self.mode]

        obj_paths = data["obj"]

        if self.config["obj_load_count"] != -1:
            obj_paths = obj_paths[:self.config["obj_load_count"]]
            #obj_paths = obj_paths[10:14]

        return obj_paths

    def _load_all_from_disk(self, uv_unwrap=True):
        self.objects = [self._load_obj(file, uv_unwrap=uv_unwrap) for file in self.obj_paths]

    def _load_all_from_scene(self):
        other_scene = bpy.path.abspath("//template_scene_import.blend")

        if not os.path.isfile(other_scene):
            other_scene = "/storage/slurm/galim/Documents/master-thesis/scenes/template_scene_import.blend"

        # name of object(s) to append or link
        obj_names = [ObjectPool.obj_name + "_" + file for file in self.obj_paths]

        with bpy.data.libraries.load(other_scene, link=False) as (data_from, data_to):
            for obj_name in obj_names:
                if obj_name in data_from.objects:
                    data_to.objects.append(obj_name)
                else:
                    raise Exception(obj_name, "not in", other_scene)

        for obj in data_to.objects:
            if obj is not None:
                bpy.context.scene.objects.link(obj)

        self.objects = [bpy.data.objects[n] for n in obj_names]

    """
    def _load_all_from_scene(self):
        self.objects = [bpy.data.objects[self.obj_name + "_" + file] for file in self.obj_paths]
    """

    def _load_obj(self, file, uv_unwrap=True):
        bpy.ops.import_mesh.stl(filepath=os.path.join(self.scene_pool_path, file), global_scale=1e-2)
        obj = bpy.context.selected_objects[0]

        bpy.context.scene.objects.active = obj
        obj.name = self.obj_name + "_" + file

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        if uv_unwrap:
            bpy.ops.object.editmode_toggle()
            bpy.ops.uv.smart_project()
            bpy.ops.object.editmode_toggle()

        _set_obj_visible(obj, False)

        return obj

    def _reset_obj(self, obj):
        _set_obj_visible(obj, False)

    def clear(self):
        if self.load_all is not None:
            for obj in self.objects:
                self._reset_obj(obj)
        else:
            for obj in self.objects:
                if obj is not None:
                    bpy.data.objects.remove(obj)

            self.objects = [None] * len(self.obj_paths)

    def select_active(self, count):
        self.clear()
        indices = random.sample(range(len(self.obj_paths)), count)
        self.active_objects = [self.get_from_all(idx) for idx in indices]

        for obj in self.active_objects:
            _set_obj_visible(obj, True)

        # print(self.active_objects)

    def get_from_all(self, idx):
        if self.objects[idx] is None:
            self.objects[idx] = self._load_obj(self.obj_paths[idx])

        return self.objects[idx]

    def __getitem__(self, item):
        return self.active_objects[item]

    def __len__(self):
        return len(self.active_objects)

    def pool_size(self):
        return len(self.obj_paths)


class SceneGen:
    mat_name = "gen_mat"

    def __init__(self, config, mode="train", mt_root=None):
        self.scene_pool_path = os.path.join(mt_root, "scene_pool") \
            if mt_root is not None \
            else "/home/kevin/Documents/master-thesis/scene_pool"

        self.mode = mode

        self.config = config

        self.tex_pool = None
        self.env_pool = None

        self.obj_pool = None

        self.world = bpy.data.worlds['World']
        self.logger = None

        self.keyframes = None

        self.focus_range = None
        self.spawn_dist_range = None
        self.f_number = None

    def set_logger(self, logger):
        self.logger = logger

    def transform_object(self, obj, trans=None, rot=None, dims=None):
        if trans is not None:
            obj.location = trans
        if rot is not None:
            obj.rotation_euler = rot

        if dims is not None:
            scale = dims / (np.linalg.norm(obj.dimensions) / obj.scale[0])
            obj.scale = scale

    # [r, theta, phi}
    # r is distance
    # theta goes up direction
    # phi goes to right
    def polar_coords(self, pos):
        r, theta, phi = pos

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return np.array([x, y, z])

    def get_camera_fov2(self):
        aspect_ratio = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y

        focal_length = bpy.data.cameras["Camera"].lens  # in mm

        sensor_width = bpy.data.cameras["Camera"].sensor_width  # in mm
        sensor_height = sensor_width / aspect_ratio  # in mm

        fov2_x = lens_tools.calc_fov(focal_length, sensor_width) / 2
        fov2_y = lens_tools.calc_fov(focal_length, sensor_height) / 2

        return fov2_x, fov2_y

    def random_transrot(self, fov2_x, fov2_y):
        base_pos = np.array([0.0, 0.5 * np.pi, 0.5 * np.pi])

        pos_offset = np.random.uniform(
            [self.spawn_dist_range[0], -fov2_y, -fov2_x],
            [self.spawn_dist_range[1], fov2_y, fov2_x]
        )

        pos = self.polar_coords(base_pos + pos_offset)
        rot = np.random.uniform(0, 2*np.pi, 3)

        return pos, rot

    def get_closest_dist_to_cam(self, obj, euclid_dist=True):
        positions = [self.keyframes[i][obj][0] for i in [0, -1]]
        dist_func = (lambda x: np.linalg.norm(x)) if euclid_dist else (lambda x: x.location.y)

        return min(dist_func(p) for p in positions)

    def random_dimensions(self, obj=None):
        dist_to_cam = self.get_closest_dist_to_cam(obj)

        dims = [random.uniform(self.config["scale_range"][0], self.config["scale_range"][1])
                * (dist_to_cam - self.config.get("near_limit", 0))] * 3

        return dims

    def import_tex(self, tex_dir, name, tex_type="diff"):
        if os.path.isdir(tex_dir):
            file = os.path.join(tex_dir, tex_type + ".jpg")
        elif tex_type == "diff":
            file = tex_dir
        else:
            file = None

        if file is not None:
            if os.path.isfile(file):
                tex_name = name + "_" + tex_type
                print(tex_name)

                img = bpy.data.images.load(file, check_existing=True)
                img.name = tex_name

                return img
            else:
                print(file, "not existing")
                return None

    def create_tex_pool(self, tex_dirs, tex_type="tex"):
        if tex_type == "tex":
            return {
                os.path.split(d)[1]: {
                    "diff": self.import_tex(d, os.path.split(d)[1], "diff"),
                    "spec": self.import_tex(d, os.path.split(d)[1], "spec")
                }
                for d in tex_dirs
            }
        elif tex_type == "env":
            return [self.import_tex(d, os.path.split(d)[1]) for d in tex_dirs]

    def create_tex_node_input(self, mat, img, next_node_input, name=None):
        diff_tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        diff_tex_node.image = img
        mat.node_tree.links.new(next_node_input, diff_tex_node.outputs[0])

        if name is not None:
            diff_tex_node.name = name

    def create_noise_tex_node(self, mat, next_node_input):
        noise_node = mat.node_tree.nodes.new("ShaderNodeTexNoise")
        noise_node.inputs[1].default_value = np.random.uniform(-5, 5)  # scale
        noise_node.inputs[2].default_value = np.random.uniform(-5, 5)  # detail
        noise_node.inputs[3].default_value = np.random.uniform(-5, 5)  # distortion

        mat.node_tree.links.new(next_node_input, noise_node.outputs[0])

    def is_texture(self, obj):
        return isinstance(obj, bpy.types.Image)

    def set_tex_or_value(self, mat, bsdf_input, img_value):
        if self.is_texture(img_value):
            self.create_tex_node_input(mat, img_value, bsdf_input)
        elif img_value == "noise":
            self.create_noise_tex_node(mat, bsdf_input)
        elif img_value is not None:
            bsdf_input.default_value = img_value

    def create_material(self, diff, spec, diff_rough=0, spec_rough=0.2, mix=0.5):
        template_mat = bpy.data.materials["template_mat"]

        mat = template_mat.copy()
        mat.name = self.mat_name + ".000"

        diff_bsdf, spec_bsdf = mat.node_tree.nodes["Diffuse BSDF"], mat.node_tree.nodes["Glossy BSDF"]
        mix_shader = mat.node_tree.nodes["Mix Shader"]

        self.set_tex_or_value(mat, diff_bsdf.inputs[0], diff)
        self.set_tex_or_value(mat, spec_bsdf.inputs[0], spec)

        self.set_tex_or_value(mat, diff_bsdf.inputs[1], diff_rough)
        self.set_tex_or_value(mat, spec_bsdf.inputs[1], spec_rough)

        mix_shader.inputs[0].default_value = mix  # mix if not (self.is_texture(diff) and not self.is_texture(spec)) else 0

        return mat

    def random_tex_or_color(self):
        tex_coll = [None, None]

        rand = random.random()

        if rand < self.config["tex_prop"]:
            k = random.choice(list(self.tex_pool.keys()))
            tex_coll = [self.tex_pool[k]["diff"], self.tex_pool[k]["spec"]]

            if tex_coll[1] is None:
                tex_coll[1] = np.concatenate([np.random.uniform(0, 1, 3), [1]])
        elif rand < self.config["tex_prop"] + self.config["noise_tex_prop"]:
            for i in range(len(tex_coll)):
                tex_coll[i] = "noise"
        else:
            for i in range(len(tex_coll)):
                tex_coll[i] = np.concatenate([np.random.uniform(0, 1, 3), [1]])

        return tex_coll

    """
    def random_obj(self):
        diff, spec = self.random_color()
        diff_rough, spec_rough, mix = np.random.uniform(0, 1, 3)

        mat = self.create_material(diff, spec, diff_rough=diff_rough, spec_rough=spec_rough, mix=mix)

        file = random.choice(self.obj_paths)
        return self.import_obj(file, mat=mat)
    """

    def randomize_obj(self, obj):
        diff, spec = self.random_tex_or_color()
        diff_rough, spec_rough, mix = np.random.uniform(0, 1, 3)

        mat = self.create_material(diff, spec, diff_rough=diff_rough, spec_rough=spec_rough, mix=mix)

        obj.active_material = mat

    def clear_data(self):
        for img in bpy.data.images:
            if img.name != "env":
                bpy.data.images.remove(img)

        self.clear_mat()

    def clear_mat(self):
        for mat in bpy.data.materials:
            if mat.name.startswith(SceneGen.mat_name):
                bpy.data.materials.remove(mat)

    def obj_set_key(self, obj, idx, pos, rot):
        obj.location = pos
        obj.rotation_euler = rot

        for key_type in ["location", "rotation_euler"]:
            obj.keyframe_insert(
                data_path=key_type,
                frame=idx)

    def obj_get_posrot(self, obj):
        return obj.location, mathutils.Vector(obj.rotation_euler)

    def set_keyframes(self, keyframes):
        if self.logger is not None:
            self.logger.info("Request setting of {} keyframes for {} objects".format(
                len(keyframes), len(keyframes[0])
            ))

        for t, keyframe in enumerate(keyframes):
            for obj, (pos, rot) in keyframe.items():
                self.obj_set_key(obj, t, pos, rot)

    def clear_keyframes(self):
        for obj in self.obj_pool:
            for t in range(self.config["num_frames"]):
                for key_type in ["location", "rotation_euler"]:
                    obj.keyframe_delete(
                        data_path=key_type,
                        frame=t)

    def animate_obj(self, obj, timescale=1):
        move_speed = mathutils.Vector([np.random.uniform(*self.config["move_speed_range"]), 0, 0])
        move_speed.rotate(mathutils.Euler(np.random.uniform(0, 2*np.pi, 3)))

        rot_speed = mathutils.Vector(np.random.uniform(-self.config["max_rot_speed"], self.config["max_rot_speed"], 3))

        pos, rot = self.obj_get_posrot(obj)

        obj_keys = [
            (pos + timescale * t * move_speed, rot + timescale * t * rot_speed)
            for t in range(self.config["num_frames"])
        ]

        # self.obj_set_keys(obj, obj_keys)

        return obj_keys

    def create_scene(self):
        print("Num obj files", self.obj_pool.pool_size())

        spawn_count = self.config["spawn_count"]

        self.lens_setup()

        if isinstance(spawn_count, int):
            spawn_count = [spawn_count] * 2

        self.obj_pool.select_active(random.randint(*spawn_count))

        self.clear_mat()

        for obj in self.obj_pool:
            self.randomize_obj(obj)

    def shuffle_obj_transrot(self):
        fov2_x, fov2_y = self.get_camera_fov2()
        for obj in self.obj_pool:
            pos, rot = self.random_transrot(fov2_x, fov2_y)
            self.transform_object(obj, trans=pos, rot=rot)

    def shuffle_obj_dim(self):
        for obj in self.obj_pool:
            dims = self.random_dimensions(obj)
            self.transform_object(obj, dims=dims)

    # keyframes [keyframe0: [(obja_pos, obja_rot), ...], ...]
    def animate_scene(self, timescale=1):
        self.keyframes = None

        for obj in self.obj_pool:
            obj_seq = self.animate_obj(obj, timescale=timescale)

            if self.keyframes is None:
                self.keyframes = [{} for _ in range(len(obj_seq))]

            for i, obj_key in enumerate(obj_seq):
                self.keyframes[i][obj] = obj_key

        self.set_keyframes(self.keyframes)

    def get_animated_keyframes(self):
        return self.keyframes

    def get_data_paths(self):
        with open(os.path.join(self.scene_pool_path, "split.json"), "r") as f:
            data = json.load(f)[self.mode]

        tex_paths = [os.path.join(self.scene_pool_path, p) for p in data["tex"]]
        env_paths = [os.path.join(self.scene_pool_path, p) for p in data["env"]]

        return tex_paths, env_paths

    def get_focus_range(self):
        return self.focus_range

    def get_fnumber(self):
        return self.f_number

    def _get_target_aperture_coc(self):
        focal_length = 4.2 * 1e-3
        focus_distance = 0.1  # 0.1
        f_number = 1.7
        depth = 0.5
        # sensor_size = 5.645 * 1e-3

        aperture = focal_length / f_number
        signed_coc = lens_tools.calc_signed_coc(focal_length, focus_distance, f_number, depth)

        return aperture, signed_coc

    def get_config(self):
        config = self.config.copy()

        fov2_x, fov2_y = self.get_camera_fov2()

        config["fov"] = 2 * fov2_x, 2 * fov2_y
        config["fov_deg"] = np.rad2deg(2 * fov2_x), np.rad2deg(2 * fov2_y)

        if "focal_length_range" in self.config:
            config["focus_range"] = self.focus_range
            config["spawn_dist_range"] = self.spawn_dist_range
            config["f_number"] = self.f_number
            config["focal_length"] = bpy.data.cameras["Camera"].lens

        return config

    def lens_setup(self):
        if "focal_length_range" in self.config:
            assert "focus_range" not in self.config
            assert "spawn_dist_range" not in self.config
            assert "f_number" not in self.config
            assert "focal_length" not in self.config

            target_aperture, target_coc = self._get_target_aperture_coc()

            focal_length = np.random.uniform(*self.config["focal_length_range"])

            self.f_number = 1e-3 * focal_length / target_aperture

            focus_multiplier = np.random.uniform(*self.config["focus_multiplier_range"])
            self.focus_range = lens_tools.calc_focus_distance_for_scale_range(
                1e-3 * focal_length, self.f_number, target_coc, focus_multiplier
            )

            self.spawn_dist_range = self.focus_range[0], self.focus_range[0] * focus_multiplier * 2
        else:
            self.focus_range = self.config["focus_range"]
            self.spawn_dist_range = self.config["spawn_dist_range"]
            self.f_number = self.config["f_number"]
            focal_length = self.config["focal_length"]

        bpy.context.scene.render.resolution_x = self.config["resolution"][0]
        bpy.context.scene.render.resolution_y = self.config["resolution"][1]
        bpy.data.cameras["Camera"].lens = focal_length

        bpy.data.cameras["Camera"].sensor_width = self.config["sensor_dim"][0]
        bpy.data.cameras["Camera"].sensor_height = self.config["sensor_dim"][1]
        bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"

    def set_light(self, pos, rot, size, strength, color):
        obj = bpy.data.objects["Light"]
        light = bpy.data.lamps["Light"]

        if pos is not None:
            obj.location = pos

        if rot is not None:
            obj.rotation_euler = rot

        if size is not None:
            light.size = size

        if color is not None:
            light.node_tree.nodes["Emission"].inputs[0].default_value = color

        if strength is not None:
            light.node_tree.nodes["Emission"].inputs[1].default_value = strength

    def random_light(self):
        if self.config.get("use_rand_light_pos", False):
            dist_range = self.config.get("light_dist_range", [1.5, 2.5])
            size_range = self.config.get("light_size_range", [4, 6])
            strength_range = self.config.get("light_strenth_range", [100, 2000])
            light_rot_pert_range = self.config.get("light_rot_pert_range", [-0.5, 0.5])

            r, theta, phi = np.random.uniform([0, 0, dist_range[0]], [2*np.pi, 2*np.pi, dist_range[1]])

            pos = [r, theta, phi]

            pos = self.polar_coords(pos)
            rot = [0, theta + np.random.uniform(*light_rot_pert_range), phi + np.random.uniform(*light_rot_pert_range)]
            size = np.random.uniform(*size_range)
            strength = np.random.uniform(*strength_range)
        else:
            pos, rot, size, strength = [None] * 4

        if self.config.get("use_rand_light_color", False):
            color = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
        else:
            color = None

        self.set_light(
            pos=pos,
            rot=rot,
            size=size,
            strength=strength,
            color=color
        )

    def scene_shuffle(self):
        _set_obj_visible(bpy.data.objects["Wall"], self.config.get("use_wall", True))

        env_tex_prop = self.config.get("env_tex_prop", None)

        if env_tex_prop is not None and random.random() < env_tex_prop:
            bpy.data.worlds["World"].use_nodes = True
            self.world.node_tree.nodes["EnvTex"].image = random.choice(self.env_pool)
        else:
            bpy.data.worlds["World"].use_nodes = False

        self.random_light()

        self.create_scene()
        self.shuffle_obj_transrot()
        self.animate_scene()
        self.shuffle_obj_dim()
        bpy.context.scene.frame_set(0)

        print("Done", "obj", self.obj_pool.pool_size(), "tex", len(self.tex_pool))

    def scene_init(self, load_all_obj=None):
        self.clear_data()

        tex_paths, env_paths = self.get_data_paths()

        self.tex_pool = self.create_tex_pool(tex_paths)
        self.env_pool = self.create_tex_pool(env_paths, tex_type="env")

        print("Num tex files", len(self.tex_pool))
        print("Num env files", len(self.env_pool))

        self.obj_pool = ObjectPool(self.scene_pool_path, self.mode, config=self.config, load_all=load_all_obj)


def scene_gen_test():
    with open("/home/kevin/Documents/master-thesis/code/mt-project/blender/scene_cfg/s7_test_moreobj.json", "r") as f:
        scene_config_file = json.load(f)

    scene_gen = SceneGen(mode="train", config=scene_config_file)

    s1 = time.time()
    #scene_gen.scene_init(load_all_obj="scene")
    scene_gen.scene_init(load_all_obj="scene")
    s2 = time.time()
    scene_gen.scene_shuffle()
    s3 = time.time()

    print("Init:", s2 - s1, "s")
    print("Suffle:", s3 - s2, "s")
    print("Total:", s3 - s1, "s")


def import_test():
    # path to the blend
    filepath = "/home/kevin/Documents/master-thesis/scenes/template_scene_import.blend"

    # name of object(s) to append or link
    obj_names = [ObjectPool.obj_name + "_" + "Thingi10K/raw_meshes/289652.stl"]

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        for obj_name in obj_names:
            if obj_name in data_from.objects:
                data_to.objects.append(obj_name)
            else:
                raise Exception(obj_name, "not in", filepath)

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.scene.objects.link(obj)


def run():
    scene_gen_test()
    #import_test()


bl_info = {
    "name": "Generate Random Scene",
    "category": "Object",
}


class SceneGenStore:
    scene_gen = None


class SceneGenOperator(bpy.types.Operator):
    bl_idname = "object.scene_gen"
    bl_label = bl_info["name"]
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if SceneGenStore.scene_gen is None:
            SceneGenStore.scene_gen = SceneGen()
            SceneGenStore.scene_gen.scene_init(load_all_obj=True)

        # SceneGenStore.scene_gen.clear_data()
        # SceneGenStore.scene_gen.clear_mat()

        SceneGenStore.scene_gen.scene_shuffle()

        return {'FINISHED'}


def register():
    bpy.utils.register_class(SceneGenOperator)


def unregister():
    bpy.utils.unregister_class(SceneGenOperator)


def main():
    run()
    #register()


if __name__ == "__main__":
    main()
