import os
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import random
import json
import torch
import torchvision
import cv2

from data import dataset, data_transforms
from tools.tools import load_exr, load_blender_flow_exr, json_support, natural_sort, torch_expand_back_as, \
    is_size_equal, is_size_greater, is_size_less
from tools.camera_lens import CameraLens
from data.data_chunk import FrameSeqChunk, FrameTypeSeqChunkConst
from tools.vis_tools import plot_tensor_grid


# Dining room min max
# old Min val: 20433 ~ 0.311
# old Max val: 48044 ~ 0.734

# Min val: 20472 ~ 0.312
# Max val: 49669 ~ 0.758


@json_support
class VideoDepthFocusData(dataset.Dataset):
    use_config = True
    crop_size = 256

    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name):
        super().__init__(root_dir, data_type, data_folder_name)

        self.clip_dirs = self._get_clip_dirs()
        assert len(self.clip_dirs) > 0

        print("Data dirs", self.clip_dirs[:3] if len(self.clip_dirs) > 3 else self.clip_dirs)

        self.img_ext = None

        for ext in ["tif", "jpg"]:
            if len(glob.glob(os.path.join(self.clip_dirs[0], "color*." + ext))) > 0:
                self.img_ext = ext

        # assert self.img_ext is not None
        print("Ext", self.img_ext)

        # self.color_basename = "color{:0>4d}." + self.img_ext
        self.depth_basename_tiff = "depth{:0>4d}.tif"
        self.depth_basename_exr = "depth{:0>4d}.exr"
        self.flow_basename = "flow{:0>4d}.exr"
        self.allinfocus_basename = "allinfocus{:0>4d}.tif"

        #self.depth_normalize = data_transforms.Normalize(normalize[0], normalize[1])
        self.depth_normalize = None
        self.coc_normalize = None
        self.signed_coc_normalize = None
        self.focus_dist_normalize = None

        #print(self.frames_per_clip, self.clip_dirs)

        self.sample_count = None
        self.sample_skip = None

        self.crop_border = None
        self.rand_crop = False
        self.rand_flip = False
        self.rand_reverse = False
        self.depth_output_indices = False

        self.include_flow = False
        self.include_coc = None
        self.include_fgbg = None

        self.test_crop = False

        self.use_allinfocus = False

        self.color_noise_stddev = None
        self.depth_noise_stddev = None

        self.setting_config = None
        self.global_lens = None
        self.resolution = None
        self.num_ramps = 1
        self.ramp_length = "all"
        self.ramp_sample_count = None
        self.test_target_frame = None

        self._load_config()

        self.exr_depth = (len(glob.glob(os.path.join(self.clip_dirs[0], "depth*.tif"))) == 0) \
            if len(self.clip_dirs) > 0 else "error"

        #self.stats_path = os.path.join(root_dir, "art_scene", "stats_total.json")
        self.stats_path = os.path.join(self.data_path, "stats_total.json")

        self.limit_data = None

        self.test_single_img_seq = None
        self.target_indices = None

        self.select_focus_dists = None

        self.fixed_ramp_idx = None
        self.fixed_frame_indices = None
        self.relative_fixed_frame_indices = False

        self.select_rel_indices = None
        self.include_all_coc = False

        self.pad_to_multiple = None
        self.pad_center = False

    def configure(self,
                  sample_count=10, sample_skip=0, depth_output_indices=None,
                  rand_reverse=True, include_flow=False, include_coc=None,
                  use_allinfocus=False, color_noise_stddev=None, depth_noise_stddev=None,
                  test_crop=None, test_target_frame=None, limit_data=None, test_single_img_seq=None,
                  include_fgbg=None, rand_crop=True, crop_border=None, rand_flip=True,
                  target_indices=None, select_focus_dists=None, ramp_sample_count=1,
                  fixed_ramp_idx=None, fixed_frame_indices=None, relative_fixed_frame_indices=False,
                  select_rel_indices=None, include_all_coc=False, pad_to_multiple=32, pad_center=True
                  ):
        assert depth_output_indices is not None

        if select_focus_dists is None:
            print("Warning: select_focus_dists is None")
            #raise Exception("select_focus_dists is None wrong")
            #select_focus_dists = [0.1, 0.13, 0.2167, 0.45]

        self.sample_count = sample_count
        self.sample_skip = sample_skip
        self.crop_border = crop_border  # 50, 50
        self.rand_crop = rand_crop
        self.rand_flip = rand_flip
        self.rand_reverse = rand_reverse

        self.depth_output_indices = depth_output_indices \
            if depth_output_indices is None or isinstance(depth_output_indices, list) \
            else [depth_output_indices]

        self.include_flow = include_flow
        self.include_coc = include_coc
        self.include_fgbg = include_fgbg
        self.use_allinfocus = use_allinfocus
        self.color_noise_stddev = color_noise_stddev
        self.depth_noise_stddev = depth_noise_stddev
        self.test_crop = test_crop
        self.test_target_frame = test_target_frame
        self.limit_data = limit_data
        self.target_indices = target_indices
        self.select_focus_dists = select_focus_dists
        self.ramp_sample_count = ramp_sample_count
        self.fixed_ramp_idx = fixed_ramp_idx
        self.fixed_frame_indices = fixed_frame_indices
        self.relative_fixed_frame_indices = relative_fixed_frame_indices
        self.include_all_coc = include_all_coc
        self.pad_to_multiple = pad_to_multiple
        self.pad_center = pad_center

        if self.select_rel_indices is not None:
            self.sample_count = len(self.select_rel_indices)
        elif self.select_focus_dists is not None:
            self.sample_count = len(self.select_focus_dists)

        self.test_single_img_seq = test_single_img_seq

        self.select_rel_indices = select_rel_indices

        if self.data_type == "test":
            self.rand_crop = False
            self.rand_reverse = False
            self.rand_flip = False

        depth_scale = 1 if self.exr_depth else 10

        assert depth_scale == 1
        # self.lens.depth_scale = depth_scale

        self._get_norm_data()

        # print("Lens:", self.lens.__dict__)

        assert self.test_target_frame is None or self.test_target_frame < self.ramp_length

    @staticmethod
    def _load_lens(param_dict):
        if "focal_length" in param_dict:
            return CameraLens(
                    focal_length=param_dict["focal_length"] / 1000,
                    aperture_diameter=param_dict["aperture_size"] / 1000 if "aperture_size" in param_dict else None,
                    f_number=param_dict.get("f_number", None),
                    sensor_size_full=np.array(param_dict["sensor_dim"]) / 1000 if "sensor_dim" in param_dict else None,
                    resolution=param_dict.get("resolution", None)
                )
        else:
            return None

    @staticmethod
    def _load_config_file(subpath):
        config_file = os.path.join(subpath, "config.json")

        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                return json.load(f)
        else:
            return None

    def _load_config(self):
        self.setting_config = VideoDepthFocusData._load_config_file(self.data_path)

        if self.setting_config is not None:
            self.resolution = self.setting_config["resolution"]

            self.global_lens = VideoDepthFocusData._load_lens(self.setting_config)

            assert "focus_ramp_length" in self.setting_config

            if "focus_ramp_length" in self.setting_config:
                print("Multi ramp")
                self.ramp_length = self.setting_config["focus_ramp_length"]
                self.num_ramps = self.setting_config["num_frames"] // self.ramp_length
                print("Ramps in dataset", self.num_ramps)

                # assert self.frames_per_clip % self.ramp_length == 0
            else:
                self.ramp_length, self.num_ramps = "all", 1
        else:
            msg = "No config found"

            if self.data_type == "test":
                print(msg)
            else:
                raise Exception(msg)

    def _get_norm_data(self):
        if self.use_config:
            print("Using stats:", self.stats_path)

            with open(self.stats_path, "r") as f:
                stats = json.load(f)

            print("Data Stats", stats)

            if stats["frames"]["depth"]["min"] > 0:
                depth_min = stats["frames"]["depth"]["min"]
                print("Using stats min depth", depth_min)
            else:
                depth_min = 0.03
                print("Using hardcoded min depth 0.03!!!")

            depth_max = stats["frames"]["depth"]["max"]

            # if not self.exr_depth:
            #    depth_min = depth_min / 0xffff
            #    depth_max = depth_max / 0xffff

            focus_min, focus_max = stats["focus_min"], stats["focus_max"]

            self.focus_dist_normalize = data_transforms.Normalize(focus_min, focus_max)
            self.depth_normalize = data_transforms.Normalize(depth_min, depth_max)

            self.signed_coc_normalize, self.coc_normalize = None, None

            if self.include_coc == "signed":
                self.signed_coc_normalize = data_transforms.Normalize(
                    self.global_lens.get_signed_coc(focus_max, depth_min),
                    self.global_lens.get_signed_coc(focus_min, depth_max)
                )
            elif self.include_coc:
                if self.global_lens is not None:
                    self.coc_normalize = data_transforms.Normalize(0, self.global_lens.get_coc(focus_min, depth_max))
                else:
                    if "coc" in stats["frames"]:
                        self.coc_normalize = data_transforms.Normalize(0, stats["frames"]["coc"]["max"])
                    elif self.data_type == "test":
                        self.coc_normalize = data_transforms.Identity()
                    else:
                        raise Exception("No CoC Info")

            print("depth_normalize:", self.depth_normalize)
            print("coc_normalize:", self.coc_normalize)
            print("signed_coc_normalize:", self.signed_coc_normalize)

    def _get_clip_dirs(self):
        split_config_path = os.path.join(self.data_path, "split.json")

        sub_data_dir = "train" if self.data_type != "test" else "test"
        print("Dataset type:", self.data_type)

        if os.path.exists(split_config_path) and self.data_type != "test": #self.use_config:
            with open(split_config_path, "r") as f:
                dirs = json.load(f)[self.data_type]

            data_dirs = [os.path.join(self.data_path, sub_data_dir, d) for d in dirs]
        else:
            if self.data_type == "test":
                print("Test mode")
            else:
                print("No config found -> Use all")
            data_dirs = glob.glob(os.path.join(self.data_path, sub_data_dir, "*"))

        data_dirs_filtered = [d for d in data_dirs if os.path.exists(os.path.join(d, "params.json"))]
        data_dirs_filtered = natural_sort(data_dirs_filtered)

        print("\n".join("{} left out.".format(d) for d in data_dirs if d not in data_dirs_filtered))

        return data_dirs_filtered

    def rev_transform(self, input_data, target_data, output_data):
        rev_normalize = False
        if rev_normalize:
            return [
                input_data,
                self.depth_normalize.rev(target_data),
                self.depth_normalize.rev(output_data)
            ]
        else:
            return [
                input_data,
                target_data,
                output_data
            ]

    def get_transform(self, key, input_data, target_data):
        raise NotImplementedError

    def get_input_resolution(self):
        return self.resolution

        """
        if self.data_type == "test":
            print("yes")
            return [480, 640]
        else:
            print("no")
            return self.resolution
        """

    def transform(self, key, data):
        # depth_indices = range(len(data)) if self.depth_output_indices is None else [self.depth_output_indices]

        data_transformed = {}

        input_res = data["color"][0].size if not isinstance(data["color"][0], np.ndarray) else \
            [data["color"][0].shape[1], data["color"][0].shape[0]]

        # data_transformed["org_size"] = torch.Tensor(input_res)
        data_transformed["org_res"] = torch.tensor(
            input_res,
            dtype=torch.int32)
        # print(data_transformed["org_res"])

        tensor_crop = None

        # if self.test_crop is not None:
        #     tensor_crop = data_transforms.FiveCrop(VideoDepthFocusData.crop_size, self.test_crop)

        # print(type(data["color"][0]), input_res)

        if self.data_type == "test":
            if is_size_equal(self.crop_size, input_res):
                tensor_crop = data_transforms.Identity()
                print("Identity")
            elif is_size_greater(self.crop_size, input_res):
                # tensor_crop = data_transforms.PadToMultiple(self.pad_to_multiple, self.pad_center)
                tensor_crop = data_transforms.Identity()
                print("Pad to multiple skipped -> outdated")
            else:
                tensor_crop = data_transforms.CenterCrop(self.crop_size)
                print("Crop")

            # tensor_crop = data_transforms.FiveCrop(VideoDepthFocusData.crop_size, self.test_crop)
            crop_offset_center = [0, 0]
        else:
            if self.rand_crop:
                tensor_crop = data_transforms.RandomCrop(VideoDepthFocusData.crop_size, input_res)
                crop_offset_center = tensor_crop.get_crop_offset_center()
            else:
                crop_offset_center = [0, 0]

        common_transform = \
            [data_transforms.ToFloatTensor()] + \
            ([tensor_crop] if tensor_crop is not None else []) + \
            ([data_transforms.RandomHFlipTensor()] if self.rand_flip else [])

        color_trans = transforms.Compose(common_transform)

        depth_trans = transforms.Compose(common_transform)

        flow_trans = transforms.Compose(common_transform) if self.include_flow else None

        data_transformed["color"] = torch.stack([color_trans(x) for x in data["color"]])
        data_transformed["depth"] = torch.stack([depth_trans(d) for d in data["depth"][self.depth_output_indices]])

        if self.include_flow:
            data_transformed["flow"] = torch.stack([flow_trans(x) for x in data["flow"]])

        # need to reimplement
        assert not self.include_fgbg
        assert self.include_coc != "signed"

        data_transformed["focus_dist"] = torch.Tensor([data["focus_dist"][i] for i in self.depth_output_indices]) \
            if self.depth_output_indices is not None else torch.Tensor(data["focus_dist"])

        data_transformed["focus_dist_all"] = torch.Tensor(data["focus_dist"])

        if self.include_coc:
            if not self.include_all_coc:
                depth_for_coc = data_transformed["depth"]
            else:
                # otherwise have to refactor masked loss to take input argument of which mask layer to take for depth
                # since for all coc we have mask now but we only have one depth
                assert self.depth_output_indices == [len(data["color"]) - 1]

                depth_for_coc = torch.stack([depth_trans(d) for d in data["depth"]])
                data_transformed["depth_mask"] = depth_for_coc

            data_transformed["coc"] = self.coc_normalize(
                data["lens"].get_coc(data_transformed["focus_dist"], depth_for_coc)
            )

        """
        if self.include_coc or self.include_fgbg:
            if self.depth_output_indices is not None:
                focus_dists = torch.Tensor([data["focus_dist"][i] for i in self.depth_output_indices])
                depths = data_transformed["depth"][self.depth_output_indices]
            else:
                focus_dists = torch.Tensor(data["focus_dist"])
                depths = data_transformed["depth"]

            if self.include_coc:
                if self.include_coc == "signed":
                    data_transformed["signed_coc"] = self.signed_coc_normalize(
                        self.lens.get_signed_coc(focus_dists, depths)
                    )
                else:
                    data_transformed["coc"] = self.coc_normalize(
                        self.lens.get_coc(focus_dists, depths)
                    )

                # print(data_transformed["coc"].shape)

            if self.include_fgbg:
                data_transformed["fgbg"] = (depths > torch_expand_back_as(focus_dists, depths)).float()
        """

        # IMPORTANT!!! normalize after coc calculation
        if self.depth_normalize is not None:
            data_transformed["depth"] = self.depth_normalize(data_transformed["depth"])

            if "depth_mask" in data_transformed:
                data_transformed["depth_mask"] = self.depth_normalize(data_transformed["depth_mask"])

        # TODO: noise
        if self.color_noise_stddev is not None:
            data_transformed["color"] = data_transforms.RandomNoise(
                stddev=self.color_noise_stddev)(data_transformed["color"])

        # data_transforms.RandomNoise(stddev=self.color_noise_stddev))

        #print(data_transformed)

        """
        if key == 0:
            torchvision.utils.save_image(
                torchvision.utils.make_grid(data_transformed["color"], nrow=20),
                "/home/kevin/log.jpg")
        """

        #print("Focus dist:", data["focus_dist"])

        for label in ["depth", "coc", "signed_coc", "fgbg", "flow"]:
            if label in data_transformed:
                data_transformed[label] = VideoDepthFocusData._reduce_single_frame_tensor_dim(data_transformed[label])

        data_transformed["lens"] = data["lens"].to_torch()
        data_transformed["crop_offset_center"] = torch.tensor(crop_offset_center)

        return data_transformed

    @staticmethod
    def _reduce_single_frame_tensor_dim(x):
        return x[0] if len(x.shape) == 4 and x.shape[0] == 1 else x

    @staticmethod
    def _get_random_range_in_range(inner_range_length, inner_range_stride, outer_range_length):
        frame_indices = np.arange(inner_range_length) * inner_range_stride

        range_limit = outer_range_length - frame_indices[-1] - 1
        start_idx = random.randint(0, range_limit)
        frame_indices += start_idx

        return frame_indices

    @staticmethod
    def _get_range_for_predict_target(inner_range_length, inner_range_stride, frame_to_predict, model_output_pos):
        frame_indices = np.arange(inner_range_length) * inner_range_stride

        start_idx = frame_to_predict - frame_indices[model_output_pos]
        frame_indices += start_idx

        return frame_indices

    @staticmethod
    def _check_range_in_range(inner_range, outer_range):
        if inner_range[0] < outer_range[0] or inner_range[-1] > outer_range[-1]:
            raise Exception(f"Error {inner_range} not in ({outer_range[0]}, {outer_range[-1]})")

    def _choose_ramp_frame_indices(self, ramp_idx, key, params):
        if self.select_rel_indices is not None:
            frame_indices = np.round(np.array(self.select_rel_indices) * (self.ramp_length - 1)).astype(int)
        elif self.select_focus_dists is not None:
            data_focus_dists = self._get_all_focus_dists(params)
            data_focus_dists = data_focus_dists[ramp_idx * self.ramp_length:(ramp_idx + 1) * self.ramp_length]

            frame_indices = self._match_focus_dists(
                dataset_dists=data_focus_dists,
                target_dists=self.select_focus_dists
            )

            # TODO: rand reverse
        else:
            if self.data_type != "test":
                frame_indices = VideoDepthFocusData._get_random_range_in_range(
                    self.sample_count,
                    self.sample_skip + 1,
                    self.ramp_length
                )

                # reverse randomly
                if self.rand_reverse and random.random() > 0.5:
                    frame_indices = np.flip(frame_indices)
            else:
                frame_indices = VideoDepthFocusData._get_range_for_predict_target(
                    self.sample_count,
                    self.sample_skip + 1,
                    frame_to_predict=self.test_target_frame,
                    model_output_pos=self.depth_output_indices
                )

                # frame_to_predict=self.frames_per_clip // 2 if self.test_target_frame is None else self.test_target_frame
                # model_output_pos=self.depth_output_indices if self.depth_output_indices is not None else self.sample_count // 2

        VideoDepthFocusData._check_range_in_range(frame_indices, range(self.ramp_length))

        frame_indices += ramp_idx * self.ramp_length

        return frame_indices

    def choose_frame_indices(self, key, params, clip_frame_count):
        if self.fixed_frame_indices is None:
            if self.fixed_ramp_idx is None:
                if self.data_type == "test":
                    ramp_indices = [0]
                else:
                    ramp_sample_count = self.ramp_sample_count if self.ramp_sample_count != "all" else self.num_ramps
                    ramp_idx_start = random.randint(0, self.num_ramps - ramp_sample_count)
                    ramp_indices = range(ramp_idx_start, ramp_idx_start + ramp_sample_count)
            else:
                ramp_indices = [self.fixed_ramp_idx]

            frame_indices = np.concatenate([
                self._choose_ramp_frame_indices(ramp_idx, key, params)
                for ramp_idx in ramp_indices
            ])

            VideoDepthFocusData._check_range_in_range(frame_indices, range(clip_frame_count))

            # duplicate test
            assert len(frame_indices) == len(set(frame_indices))

            return frame_indices
        else:
            if self.relative_fixed_frame_indices:
                print("yes")
                return np.round(np.array(self.fixed_frame_indices) * (clip_frame_count - 1)).astype(int)
            else:
                return list(self.fixed_frame_indices)

    # def _create_placeholder_params(self):
    #     return {"frames": [{"idx": i, "focDist": -1} for i in range(self.frames_per_clip)]}

    def load_item(self, key):
        params = self._load_params(key)
        clip_frame_count = len(params["frames"])

        if len(params) == 0:
            raise Exception("Params missing")
            # print("Params empty!!!")
            # params = self._create_placeholder_params()

        frame_indices = self.choose_frame_indices(key, params, clip_frame_count)
        # print(frame_indices)

        lens = VideoDepthFocusData._load_lens(params)
        if lens is None:
            lens = self.global_lens
        if lens is None:
            lens = VideoDepthFocusData._load_lens(self._load_config_file(self._clip_folder(key)))
        assert lens is not None

        data_dict = FrameSeqChunk(self._clip_folder(key), frame_indices, params, self.data_type, self.img_ext, lens)

        if self.data_type == "test":
            print(frame_indices)
            print(data_dict["focus_dist"])

        data_dict = self._modify_data(data_dict)

        return data_dict

    def _modify_data(self, data_dict):
        if self.test_single_img_seq is not None:
            target_idx = self.depth_output_indices if self.depth_output_indices is not None else self.sample_count // 2
            target_idx %= self.sample_count

            for i in range(len(data_dict["color"])):
                if i != target_idx:
                    labels = ["color"]

                    if isinstance(data_dict["depth"], list):
                        labels.append("depth")

                    for label in labels:
                        if self.test_single_img_seq == "repeat":
                            data_dict[label][i] = data_dict[label][target_idx]
                        elif self.test_single_img_seq == "zero":
                            if isinstance(data_dict["color"][target_idx], Image.Image):
                                dim = [
                                    data_dict["color"][target_idx].size[1],
                                    data_dict["color"][target_idx].size[0],
                                    1 if label == "depth" else 3
                                ]
                            else:
                                dim = [
                                    data_dict["color"][target_idx].shape[0],
                                    data_dict["color"][target_idx].shape[1],
                                    1 if label == "depth" else 3
                                ]


                            data_dict[label][i] = self._create_img(dim, dtype=np.uint8)
                        else:
                            raise Exception("No modfiy data self.test_single_img_seq ==", self.test_single_img_seq)

        return data_dict

    def _compose_data(self, key, frame_indices, params):
        pass
        """
        load_all_depth_frames = self.depth_output_indices is None or self.include_coc

        return {
            "color":
                self._load_color_range(key, frame_indices, params) if not self.use_allinfocus else
                self._load_allinfocus_range(key, frame_indices, params)
            ,
            "depth":
                self._load_depth_range(key, frame_indices, params) if load_all_depth_frames else
                self._load_depth(key, frame_indices[self.depth_output_indices], params),
            "flow":
                self._load_flow_range(key, frame_indices, params) if self.include_flow else None,
            "focus_dist": self._get_focus_dists(params, frame_indices)
        }
        """

    def _get_filename(self, clip_idx, frame_idx, params, fmt, raise_no_exist=True):
        file_idx = params["frames"][frame_idx]["idx"]

        filename = os.path.join(
            self._clip_folder(clip_idx),
            fmt.format(file_idx)
        )

        if not os.path.isfile(filename):
            if raise_no_exist:
                raise Exception(f"Error: {filename} not found")

            return None
        else:
            return filename

    def _get_all_focus_dists(self, params):
        return [f["focDist"] for f in params["frames"]]

    def _get_focus_dists(self, params, frame_indices):
        focus_dists = self._get_all_focus_dists(params)
        return [focus_dists[i] for i in frame_indices]

    def _load_color_range(self, clip_idx, frame_indices, params):
        return [self._read_img(self._get_filename(clip_idx, i, params, self.color_basename)) for i in frame_indices]

    def _load_flow_range(self, clip_idx, frame_indices, params):
        return [load_blender_flow_exr(
            self._get_filename(clip_idx, i, params, self.flow_basename)
        ) for i in frame_indices[:-1]]

    def _create_img(self, dim, val=0, dtype=None):
        return np.full(dim, val, dtype=dtype)

    def _load_allinfocus_range(self, clip_idx, frame_indices, params):
        return [self._read_img(self._get_filename(clip_idx, i, params, self.allinfocus_basename)) for i in frame_indices]

    def _read_img(self, path):
        return Image.open(path)

    def _get_closest_idx(self, arr, val):
        min_idx = np.argmin(np.abs(np.array(arr) - val))
        # if self.data_type == "test":
        #     print("{:>4d} | {:.5f} -> {:.5f}".format(min_idx, val, arr[min_idx]))
        return min_idx

    def _match_focus_dists(self, dataset_dists, target_dists):
        closest_indices = [self._get_closest_idx(dataset_dists, target_dist)
                           for target_dist in target_dists]

        return np.array(closest_indices)

    def _load_params(self, clip_idx):
        with open(os.path.join(self._clip_folder(clip_idx), "params.json"), "r") as f:
            return json.load(f)

    def _clip_folder(self, i):
        return self.clip_dirs[i]

    def __len__(self):
        return len(self.clip_dirs) if self.limit_data is None else self.limit_data


class VideoDepthFocusDataMp4(VideoDepthFocusData):
    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name):
        super().__init__(root_dir, data_type, data_folder_name)

    def _get_video_frame_by_idx(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print("Error cap.read at " + str(idx))
            return None
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_frames(self, clip_idx, frame_indices):
        video_path = os.path.join(
            self._clip_folder(clip_idx),
            "color.mp4"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening clip: {}".format(video_path))

        frames = [self._get_video_frame_by_idx(cap, i) for i in frame_indices]
        cap.release()

        if any(x is None for x in frames):
            raise Exception("Error reading clip: {}".format(video_path))

        return frames

    def load_item(self, key):
        assert self.select_focus_dists is not None

        with open(os.path.join(self._clip_folder(key), "color.json")) as f:
            config = json.load(f)

        focus_dists_video = np.array([f["focusDist"] for f in config["frames"]])

        indices = self._match_focus_dists(focus_dists_video, self.select_focus_dists)

        depth = self._create_img([self.resolution[1], self.resolution[0], 1], np.nan)

        print(indices, focus_dists_video[indices])

        return {
            "color": self._load_frames(key, indices),
            "depth": FrameTypeSeqChunkConst([depth] * len(indices)),
            "focus_dist": focus_dists_video[indices],
            "lens": self.global_lens
        }

    def _load_params(self, clip_idx):
        video_info_path = os.path.join(
            self._clip_folder(clip_idx),
            "color.json"
        )

        with open(video_info_path, "r") as f:
            video_info = json.load(f)

        return {"frames": [{"idx": i, "focDist": f["focusDist"]} for i, f in enumerate(video_info["frames"])]}


class VideoDepthFocusDataFiveCrop(dataset.DatasetJoin):
    def __init__(self, dataset_gen):
        datasets = [dataset_gen() for _ in range(5)]

        for i in range(5):
            datasets[i].test_crop = i

        super().__init__(datasets)

    @property
    def depth_output_indices(self):
        return self.datasets[0].depth_output_indices

    @property
    def sample_count(self):
        return self.datasets[0].sample_count

    @property
    def lens(self):
        return self.datasets[0].lens

    @property
    def coc_normalize(self):
        return self.datasets[0].coc_normalize

    @property
    def depth_normalize(self):
        return self.datasets[0].depth_normalize

    @property
    def signed_coc_normalize(self):
        return self.datasets[0].signed_coc_normalize
