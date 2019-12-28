import bpy
import math
import os
import json
import contextlib
import sys
from blender.log_tools import create_logger

"""
filename = "/home/kevin/Documents/master-thesis/code/blender/render.py"
exec(compile(open(filename).read(), filename, 'exec'))
"""


"""
class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
"""

@contextlib.contextmanager
def nostdout():
    # redirect output to log file
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    yield

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)


class RenderSetup:
    scene = bpy.context.scene
    node_tree = scene.node_tree

    mt_root = \
        "/storage/slurm/galim/Documents/master-thesis" \
        if os.path.isdir("/storage/slurm") else "/home/kevin/Documents/master-thesis"

    render_dir = \
        os.path.join(mt_root, "render")

    cuda_idx = 1

    def __init__(self, root_dir, config, scene_generate, job_name=None):
        self.root_dir = root_dir
        self.logger = None
        self.config = config
        self.scene_generate = scene_generate
        self.job_name = job_name

        path = None
        for i in range(999999999):
            path = self.root_dir.format(index=i)
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                    break
                except FileExistsError:
                    print("Caught FileExistsError. Trying next sequence:", i+1)

        self.root_dir = path

    def _save_config(self):
        with open(os.path.join(self.root_dir, "config.json"), "w") as f:
            json.dump(self.scene_generate.get_config(), f, indent=4)

    def start(self):
        self.logger = create_logger(os.path.join(self.root_dir, "render.log"))
        self.logger.info("--- START ---")
        self.logger.info("Job: "  + (self.job_name if self.job_name is not None else "None"))

    def apply_settings(self):
        pass

    def render(self):
        pass

    def render_indices(self, indices):
        for i in indices:
            self._set_frame(i)
            self.render()

    def render_range(self, start=0, stop=100):
        self.render_indices(range(start, stop + 1))

    def _render_scene(self, supress_stdout=False):
        if supress_stdout:
            with nostdout():
                bpy.ops.render.render()
        else:
            bpy.ops.render.render()

    def _set_cam(self):
        pass

    def _set_frame(self, i):
        self.scene.frame_set(i)

    def set_camera(self, cam_name):
        self.scene.camera = bpy.data.objects[cam_name]

    def _camera(self):
        return bpy.data.cameras["Camera"]

    def _action(self, action_name):
        return bpy.data.actions[action_name]

    def _object(self, obj_name):
        return bpy.data.objects[obj_name]

    def enable_cpu_only(self):
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "NONE"
        bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        bpy.context.user_preferences.addons['cycles'].preferences.devices[self.cuda_idx].use = False
        self.scene.cycles.device = "CPU"

    def enable_gpu(self):
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
        for i in range(len(bpy.context.user_preferences.addons['cycles'].preferences.devices)):
            bpy.context.user_preferences.addons['cycles'].preferences.devices[i].use = True
        self.scene.cycles.device = "GPU"

    def log_settings(self):
        self.logger.info(
            "Settings\n"
            "Blender version: {}\n"
            "Device: {}\n"
            "Resolution: {}x{}\n"
            "Samples: {}\n"
            "Denoising: {}\n"
            "Tiles: {}x{}\n"
            "Bounces: (Total: [{}, {}], Diffuse: {}, Glossy: {}, Transmission: {}, Transparent: [{}, {}])\n"
            "Clamp: (Dir: {}, Indir: {})\n"
            "Caustics: (Refl: {}, Refr: {})\n".format(
                bpy.app.version_string,
                bpy.context.user_preferences.addons['cycles'].preferences.devices[
                    self.cuda_idx].name if self.scene.cycles.device == "GPU"
                else "CPU",
                self.scene.render.resolution_x,
                self.scene.render.resolution_y,
                self.scene.cycles.samples,
                self.scene.render.layers[0].cycles.use_denoising,
                self.scene.render.tile_x,
                self.scene.render.tile_y,
                self.scene.cycles.min_bounces,
                self.scene.cycles.max_bounces,
                self.scene.cycles.diffuse_bounces,
                self.scene.cycles.glossy_bounces,
                self.scene.cycles.transmission_bounces,
                self.scene.cycles.transparent_min_bounces,
                self.scene.cycles.transparent_max_bounces,
                self.scene.cycles.sample_clamp_direct,
                self.scene.cycles.sample_clamp_indirect,
                self.scene.cycles.caustics_reflective,
                self.scene.cycles.caustics_refractive
            )
        )


class RenderSetupColorDepth(RenderSetup):
    supported_modes = ["performance", "medium", "quality"]

    def __init__(self, root_dir, qual_mode, sub_dir, focus_sweep_speed, use_blender_render_depth, config,
                 scene_generate, job_name):
        super().__init__(root_dir=os.path.join(root_dir, sub_dir), config=config, scene_generate=scene_generate,
                         job_name=job_name)

        self.qual_mode = qual_mode
        self.focus_sweep_speed = focus_sweep_speed

        self.use_blender_render_depth = use_blender_render_depth

        self.setup_test = False

        print("use blender render depth:", self.use_blender_render_depth)

    def start(self):
        super().start()
        self.scene_generate.set_logger(self.logger)

    def apply_render_quality_settings(self, mode):
        if mode == "performance":
            self.scene.cycles.samples = 100

            self.scene.render.layers[0].cycles.use_denoising = True

            self.scene.render.tile_x = 512
            self.scene.render.tile_y = 512

            self.scene.cycles.min_bounces = 0
            self.scene.cycles.max_bounces = 1
            self.scene.cycles.diffuse_bounces = 1
            self.scene.cycles.glossy_bounces = 1
            self.scene.cycles.transmission_bounces = 0
            self.scene.cycles.transparent_min_bounces = 0
            self.scene.cycles.transparent_max_bounces = 0

            self.scene.cycles.sample_clamp_direct = 0.5
            self.scene.cycles.sample_clamp_indirect = 0.0

            self.scene.cycles.caustics_reflective = False
            self.scene.cycles.caustics_refractive = False
        elif mode.startswith("medium"):
            if mode == "medium100":
                self.scene.cycles.samples = 100
            elif mode == "medium150":
                self.scene.cycles.samples = 150
            elif mode == "medium200":
                self.scene.cycles.samples = 200
            elif mode == "medium256":
                self.scene.cycles.samples = 256
            elif mode == "medium512":
                self.scene.cycles.samples = 512
            elif mode == "medium1024":
                self.scene.cycles.samples = 1024
            else:
                self.logger.info("Mode '{}' not recognized".format(mode))
                exit(1)

            self.scene.render.layers[0].cycles.use_denoising = True

            self.scene.render.tile_x = 512
            self.scene.render.tile_y = 512

            self.scene.cycles.min_bounces = 0
            self.scene.cycles.max_bounces = 2
            self.scene.cycles.diffuse_bounces = 2
            self.scene.cycles.glossy_bounces = 1
            self.scene.cycles.transmission_bounces = 0
            self.scene.cycles.transparent_min_bounces = 0
            self.scene.cycles.transparent_max_bounces = 0

            self.scene.cycles.sample_clamp_direct = 0.5
            self.scene.cycles.sample_clamp_indirect = 0.0

            self.scene.cycles.caustics_reflective = False
            self.scene.cycles.caustics_refractive = False
        elif mode == "quality":
            self.scene.cycles.samples = 1000

            self.scene.render.layers[0].cycles.use_denoising = True

            self.scene.render.tile_x = 512
            self.scene.render.tile_y = 512

            self.scene.cycles.min_bounces = 3
            self.scene.cycles.max_bounces = 12
            self.scene.cycles.diffuse_bounces = 4
            self.scene.cycles.glossy_bounces = 4
            self.scene.cycles.transmission_bounces = 12
            self.scene.cycles.transparent_min_bounces = 8
            self.scene.cycles.transparent_max_bounces = 8

            self.scene.cycles.sample_clamp_direct = 0.5
            self.scene.cycles.sample_clamp_indirect = 0.0

            self.scene.cycles.caustics_reflective = False
            self.scene.cycles.caustics_refractive = False
        else:
            self.logger.info("Mode '{}' not recognized".format(mode))
            exit(1)

    def _file_output_nodes(self):
        return ["OutColor", "OutDepth", "OutFlow", "OutAllinfocus"]

    def apply_settings(self):
        super().apply_settings()

        for out_node in self._file_output_nodes():
            if out_node in self.node_tree.nodes:
                self.node_tree.nodes[out_node].base_path = self.root_dir
            else:
                print(out_node, "not found")

        #self.scene.render.resolution_x = 640
        #self.scene.render.resolution_y = 480

        self.enable_gpu()
        self.apply_render_quality_settings(self.qual_mode)

        self.scene.render.tile_x = self.scene.render.resolution_x
        self.scene.render.tile_y = self.scene.render.resolution_y
        self.scene.cycles.samples = self.config["samples"] # 512

        self._camera().clip_start, self._camera().clip_end = self.config["cam_clip"]


    def apply_color_settings(self):
        self.scene.render.engine = "CYCLES"

        self.node_tree.nodes["OutColor"].mute = False
        self.node_tree.nodes["OutDepth"].mute = True
        self.node_tree.nodes["OutFlow"].mute = True

        if "OutAllinfocus" in self.node_tree.nodes:
            self.node_tree.nodes["OutAllinfocus"].mute = True

        self._camera().cycles.aperture_type = "FSTOP"
        self._camera().cycles.aperture_fstop = self.scene_generate.get_fnumber()

        """
        if "aperture_size" in self.config:
            self._camera().cycles.aperture_type = "RADIUS"
            self._camera().cycles.aperture_size = self.config["aperture_size"]
        else:
            self._camera().cycles.aperture_type = "FSTOP"
            self._camera().cycles.aperture_fstop = self.config["f_number"]
        """

    def apply_depth_settings(self):
        if self.use_blender_render_depth:
            self.scene.render.engine = "BLENDER_RENDER"
            self.scene.render.use_antialiasing = False

            if "OutAllinfocus" in self.node_tree.nodes:
                self.node_tree.nodes["OutAllinfocus"].mute = True
        else:
            self.scene.render.engine = "CYCLES"

            if "OutAllinfocus" in self.node_tree.nodes:
                self.node_tree.nodes["OutAllinfocus"].mute = False

        self.node_tree.nodes["OutColor"].mute = True
        self.node_tree.nodes["OutDepth"].mute = False
        self.node_tree.nodes["OutFlow"].mute = False

        self._camera().cycles.aperture_type = "RADIUS"
        self._camera().cycles.aperture_size = 0

    def _set_focus(self, focus_dist):
        self._camera().dof_distance = focus_dist

    def render(self):
        self.render_focus()

    def render_focus(self, foc=None):
        #self.apply_settings()

        # render all in focus part
        self.apply_depth_settings()
        self._render_scene()

        self.logger.info("Rendered Depth Pass ({})".format(
            self.scene.render.engine
        ))

        # render blur part
        self.apply_color_settings()

        if foc is not None:
            self._set_focus(foc)

        if self.setup_test:
            return

        self._render_scene()

        self.logger.info("Rendered Blur Pass (Foc Dist: {:.4f}) ({})".format(
            foc, self.scene.render.engine
        ))

    def _focus_time_func(self, t):
        t_frac = t - math.floor(t)
        return t_frac

        #t_norm = 2 * t_frac - 1
        #return -abs(t_norm) + 1

    #frame_idx in [0, self.seq_length() - 1]
    def _get_focus_dist(self, frame_idx):
        focus_min, focus_max = self.scene_generate.get_focus_range()

        focus_ramp_length = self.config.get("focus_ramp_length", self.seq_length())

        t = (frame_idx % focus_ramp_length) / (focus_ramp_length - 1)
        foc_dist = (1 - t) * focus_min + t * focus_max
        return foc_dist

    def _render_sweep(self, indices):
        self.apply_settings()
        self.log_settings()

        if indices is None:
            indices = range(0, self.frame_end() + 1)

        self.logger.info("{} started. Render: [{}]".format(
            str(self),
            indices
        ))


        params = {
            "focusRangeStart": self.scene_generate.get_focus_range()[0],
            "focusRangeEnd": self.scene_generate.get_focus_range()[1],
            "frames": []
        }

        for i in indices:
            self._set_frame(i)

            foc_dist = self._get_focus_dist(i)

            params["frames"].append({"idx": i, "focDist": foc_dist})

            self.render_focus(foc_dist)

            if self.setup_test:
                break

        self._render_flow_steps(len(indices))

        with open(os.path.join(self.root_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        self.logger.info(str(self) + " finished")

    def _render_flow_steps(self, total_frame_count):
        flow_steps = self.config.get("flow_steps", None)

        if flow_steps is not None:
            self._set_frame(0)

            self.apply_depth_settings()

            # active_nodes = ["OutFlow"]
            active_nodes = ["OutFlow", "OutDepth"]

            for node in self._file_output_nodes():
                self.node_tree.nodes[node].mute = node not in active_nodes

            scene_keys = self.scene_generate.get_animated_keyframes()

            for flow_step in flow_steps:
                for idx0 in range(0, total_frame_count - flow_step["size"], flow_step["dialation"]):
                    idx1 = idx0 + flow_step["size"]

                    # self.scene_generate.clear_keyframes()
                    # self.scene_generate.set_keyframes(scene_keys)
                    self.scene_generate.set_keyframes([scene_keys[idx0], scene_keys[idx1]])

                    for idx, direction in enumerate(["fwd", "bwd"]):
                        old_file_paths = {}

                        for node in active_nodes:
                            old_file_paths[node] = self.node_tree.nodes[node].file_slots[0].path
                            self.node_tree.nodes[node].file_slots[0].path += "_{}to{}_".format(
                                idx0, idx1
                            )

                        self._set_frame(idx)
                        self._render_scene()

                        self.logger.info("Rendered Flow Step {}to{} {}".format(
                            idx0, idx1, direction
                        ))

                        for node in active_nodes:
                            self.node_tree.nodes[node].file_slots[0].path = old_file_paths[node]

    def render_indices(self, indices):
        self._render_sweep(indices)

    def seq_length(self):
        raise NotImplementedError

    def frame_end(self):
        raise NotImplementedError
        #return self.scene.frame_end

    def __repr__(self):
        pass


# casa2
class RenderSetupDiningRoom(RenderSetupColorDepth):
    def __init__(self, qual_mode, sub_dir, config, focus_sweep_speed=1, use_blender_render_depth=False,
                 scene_generate=None, job_name=None):
        super().__init__(
            root_dir=os.path.join(self.render_dir),
            qual_mode=qual_mode, sub_dir=sub_dir,
            focus_sweep_speed=focus_sweep_speed,
            use_blender_render_depth=use_blender_render_depth,
            config=config,
            scene_generate=scene_generate,
            job_name=job_name
        )

        self.apply_settings()

    def __repr__(self):
        return "DiningRoom0917"

    def seq_length(self):
        return self.config["num_frames"]

    def frame_end(self):
        return 50000

    def render_seq(self):
        self._save_config()
        self.render_indices(range(self.seq_length()))

    """
    def render_seq(self, i):
        frame_start = i*self.seq_length()

        if frame_start < self.frame_end():
            indices = range(frame_start, frame_start + self.seq_length())
            print(indices)
            self.render_indices(indices)
            return True
        else:
            return False
    """
