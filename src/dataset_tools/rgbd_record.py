import cv2
import numpy as np
import os
import json
import time
from matplotlib import cm


class RgbdView(object):
    def __init__(self, normalize_depth=True, depth_cmap="viridis", depth_invalid_color=np.array([0, 0, 0])):
        self.normalize_depth = normalize_depth
        self.depth_cmap = depth_cmap
        self.depth_invalid_color = depth_invalid_color

        self.depthmap_fixed_scale = None # 10
        self.view_res_scale = None  # 0.5

    @staticmethod
    def u16_to_u8(img):
        return (img.astype(float) * (0xff / 0xffff)).astype(np.uint8)

    @staticmethod
    def cvt_color_to_cv(color):
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _apply_color_mask(img, color, mask=None):
        img[:, :, 0:1][mask], img[:, :, 1:2][mask], img[:, :, 2:3][mask] = color
        return img

    @staticmethod
    def apply_mpl_cmap(img, cmap):
        img_cvt = (cm.get_cmap(cmap)(img[..., 0].astype(np.float) / 0xffff)[..., :3] * 0xffff).astype(np.uint16)

        return img_cvt

    @staticmethod
    def cvt_depth_to_cv(depth, cmap, mask_color, mask, bgr_color=True):
        if cmap is None:
            depth3 = np.concatenate([depth]*3, axis=2)
        else:
            depth3 = RgbdView.apply_mpl_cmap(depth, cmap)
            if bgr_color:
                depth3 = RgbdView.cvt_color_to_cv(depth3)

        if mask_color is not None:
            depth3 = RgbdView._apply_color_mask(depth3, mask_color[[2, 1, 0]], mask)

        return depth3

    @staticmethod
    def _apply_normalize_depth(depth):
        mask_zero = depth == 0
        mask_nonzero = depth != 0

        depth_nonzero = depth[mask_nonzero]
        if depth_nonzero.size != 0:
            depth = depth.astype(float)
            depth_min, depth_max = depth_nonzero.min(), depth.max()
            depth[mask_nonzero] = (depth[mask_nonzero] - depth_min) / (depth_max - depth_min)
            depth = (depth * 0xffff).astype(np.uint16)

        return depth, mask_zero

    def process_depth(self, depth):
        if self.depthmap_fixed_scale is not None:
            depth_mask = depth == 0
            depth = (depth * self.depthmap_fixed_scale).astype(np.uint16)
        elif self.normalize_depth:
            depth, depth_mask = RgbdView._apply_normalize_depth(depth)
        elif self.depth_invalid_color is not None:
            depth_mask = depth == 0
        else:
            depth_mask = None

        depth_cv = RgbdView.cvt_depth_to_cv(depth, self.depth_cmap, self.depth_invalid_color, depth_mask)

        return depth_cv

    def show(self, color=None, depth=None, label=None, is_bgr=False):
        color_cv, depth_cv = None, None

        if color is not None:
            color_cv = RgbdView.cvt_color_to_cv(color) if not is_bgr else color

        if depth is not None:
            depth_cv = self.process_depth(depth)

        RgbdView.cv_show(color=color_cv, depth=depth_cv, label=label, view_scale=self.view_res_scale)

    @staticmethod
    def cv_show(color=None, depth=None, label=None, view_scale=None):
        frames = []

        if color is not None:
            frames.append(color)

        if depth is not None:
            #depth_cv = cv2.convertScaleAbs(depth, alpha=(255.0/65535.0))
            depth_cv = (depth * (255.0/65535.0)).astype(np.uint8)

            #if len(depth_cv.shape) == 2:
                #depth_cv = depth_cv[:, :, None]

            if depth_cv.shape[2] == 1:
                depth_cv = np.concatenate([depth_cv, depth_cv, depth_cv], axis=2)

            frames.append(depth_cv)

        view = np.concatenate(frames, axis=1)

        if view_scale is not None:
            view = cv2.resize(view,
                              dsize=(round(view.shape[1]*view_scale), round(view.shape[0]*view_scale)),
                              interpolation=cv2.INTER_NEAREST)

        if label is not None:
            cv2.putText(view, label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        cv2.imshow("RGBD View", view)

    @staticmethod
    def get_registration_view(color, depth, weight=0.5, fixed_scale=None, mask_color=np.array([0xffff, 0, 0])):
        if fixed_scale is not None:
            mask = depth == 0
            depth = (depth * fixed_scale).astype(np.uint16)
        else:
            depth, mask = RgbdView._apply_normalize_depth(depth)

        depth = RgbdView.cvt_depth_to_cv(depth, "viridis", mask_color, mask, bgr_color=False)

        depth_f, color_f = depth.astype(float), color.astype(float) * (0xffff / 0xff)

        regist = (weight * depth_f + (1 - weight) * color_f).astype(np.uint16)

        regist = RgbdView._apply_color_mask(regist, mask_color, mask)

        return regist

    def get_registration_view_inst(self, color, depth, weight=0.5):
        return RgbdView.get_registration_view(color, depth, weight=weight, fixed_scale=self.depthmap_fixed_scale,
                                              mask_color=self.depth_invalid_color)

    def save_registration_view(self, color, depth, weight=0.5):
        img_format = "png"
        regist = self.get_registration_view_inst(color, depth, weight)




    def show_registration(self, color, depth, weight=0.5):
        regist = self.get_registration_view_inst(color, depth, weight)

        cv2.imshow("registration", RgbdView.cvt_color_to_cv(regist))


class RgbdRecorderBase(object):
    def __init__(self):
        self.record_dir = os.path.join(os.path.expanduser("~"), "Pictures", "RgbdRecords")

        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

        img_format = "png"

        self.frame_color_filename = "color{}." + img_format
        self.frame_depth_filename = "depth{}." + img_format

    def is_recording(self):
        return NotImplementedError

    def start_record(self, record_color=False, record_depth=False):
        raise NotImplementedError

    def record_frame(self, color=None, depth=None):
        raise NotImplementedError

    def end_record(self):
        raise NotImplementedError

    def save_frame(self, color=None, depth=None, cvt_color=True, root_dir=None, depth_view=None):
        if root_dir is None:
            root_dir = self.record_dir

        if color is not None:
            if cvt_color:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            cv2.imwrite(
                RgbdRecorderBase.get_free_filename(root_dir, self.frame_color_filename),
                color
            )

        if depth is not None:
            if depth_view is not None:
                depth = depth_view.process_depth(depth)

            cv2.imwrite(
                RgbdRecorderBase.get_free_filename(root_dir, self.frame_depth_filename),
                depth
            )

    @staticmethod
    def get_free_filename(root_dir, file_pattern):
        for i in range(1, 1000000000):
            filename = os.path.join(root_dir, file_pattern).format(i)

            if not os.path.isfile(filename) and not os.path.isdir(filename):
                return filename


class RgbdRecorderImage(RgbdRecorderBase):
    # 2*640*480*30 / 1024^2 ~= 17.578125 MB/s
    timestamp_fmt = "%H:%M:%S.%f"

    def __init__(self):
        super().__init__()

        self.on_record = False
        self.record_start_time = -1

        self.frames_color = []
        self.frames_depth = []

        self.frame_params = None

        self.video_path_pattern = "video{}"
        self.video_path = None

    def is_recording(self):
        return self.on_record

    def start_record(self, record_color=False, record_depth=False):
        self.frames_color.clear()
        self.frames_depth.clear()

        self.frame_params = {
            "color": [],
            "depth": []
        }

        self.video_path = RgbdRecorderBase.get_free_filename(self.record_dir, self.video_path_pattern)

        os.makedirs(self.video_path)

        self.on_record = True
        self.record_start_time = time.time()

    def get_record_time(self):
        return time.time() - self.record_start_time

    def record_frame(self, color=None, depth=None, timestamp_color=None, timestamp_depth=None):
        if depth is not None:
            self.frames_depth.append(depth.copy())
            #self.frames_depth.append(depth)

        if color is not None:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            self.frames_color.append(color)

        if timestamp_color is not None:
            self.frame_params["color"].append(timestamp_color.strftime(self.timestamp_fmt))

        if timestamp_depth is not None:
            self.frame_params["depth"].append(timestamp_depth.strftime(self.timestamp_fmt))

        return color, depth

    def end_record(self):
        for color in self.frames_color:
            self.save_frame(color=color, cvt_color=False, root_dir=self.video_path)

        for depth in self.frames_depth:
            self.save_frame(depth=depth, root_dir=self.video_path)

        params_file = os.path.join(self.video_path, "params.json")
        with open(params_file, 'w') as f:
            json.dump(self.frame_params, f)

        self.frames_color.clear()
        self.frames_depth.clear()

        self.on_record = False


class RgbdRecorderVideo(RgbdRecorderBase):
    def __init__(self):
        super().__init__()

        self.video_color_filename = "color{}.avi"
        self.video_depth_filename = "depth{}.avi"

        self.video_writer_color = None
        self.video_writer_depth = None

    def is_recording(self):
        return self.video_writer_depth is not None or self.video_writer_color is not None

    def start_record(self, record_color=False, record_depth=False):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        if record_color:
            self.video_writer_color = cv2.VideoWriter(
                RgbdRecorderBase.get_free_filename(self.record_dir, self.frame_color_filename),
                fourcc,
                30.0,
                (640, 480)
            )

        if record_depth:
            self.video_writer_depth = cv2.VideoWriter(
                RgbdRecorderBase.get_free_filename(self.record_dir, self.frame_depth_filename),
                fourcc,
                30.0,
                (640, 480)
            )

    def record_frame(self, color=None, depth=None):
        if depth is not None:
            self.video_writer_depth.write(depth)

        if color is not None:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            self.video_writer_color.write(color)

        return color, depth

    def end_record(self):
        if self.video_writer_color is not None:
            self.video_writer_color.release()
            self.video_writer_color = None

        if self.video_writer_depth is not None:
            self.video_writer_depth.release()
            self.video_writer_depth = None
