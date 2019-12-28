import json
import cv2
from datetime import datetime
import os
import numpy as np
import glob


class FrameData(object):
    def __init__(self):
        self.timestamps = []

    def get_closest_frame(self, timestamp):
        ts_diff = [abs(timestamp.timestamp() - ts_ele.timestamp()) for ts_ele in self.timestamps]
        idx = np.argmin(ts_diff)
        t = self.timestamps[idx]
        timedelta = abs(timestamp - t)
        return idx, timedelta

    def get_frame_by_time(self, timestamp):
        idx, delta = self.get_closest_frame(timestamp)
        return self.get_frame_by_idx(idx), idx, delta

    def get_frame_by_idx(self, idx):
        raise NotImplementedError

    def get_frame_timestamp_by_idx(self, idx):
        return self.timestamps[idx]

    def get_timestamp(self, idx):
        #print(self.timestamps)
        #print(idx)
        return self.timestamps[idx]

    def __len__(self):
        raise NotImplementedError


class FrameDataVideo(FrameData):
    timestamp_fmt = "%H:%M:%S:%f"

    def __init__(self, root_dir, base_name=None):
        super().__init__()

        if base_name is None:
            base_name = os.path.splitext(os.path.basename(glob.glob(os.path.join(root_dir, "*"))[0]))[0]

        self.video_name = os.path.join(root_dir, base_name + ".mp4")
        self.config_name = os.path.join(root_dir, base_name + ".json")

        if os.path.isfile(self.config_name):
            with open(self.config_name, "rb") as f:
                config_data = json.load(f)
        else:
            config_data = None

        self.focus_set = []

        if config_data:
            for frame in config_data["frames"]:
                time_str = frame["time"]
                timestamp = datetime.strptime(time_str, FrameDataVideo.timestamp_fmt)
                self.timestamps.append(timestamp)
                self.focus_set.append(frame["focusDist"])

        self.cap = cv2.VideoCapture(self.video_name)

        #self.num_frames = len(config_data["frames"])
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_by_idx(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()

        if not ret:
            print("Error cap.read")
            return None
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_info_by_idx(self, idx):
        return self.timestamps[idx], self.focus_set[idx]

    def __len__(self):
        return self.num_frames


class FrameDataImage(FrameData):
    timestamp_fmt = "%H:%M:%S.%f"

    def __init__(self, root_dir, img_type="depth"):
        super().__init__()

        self.root_dir = root_dir
        self.base_name = img_type + "{}.tif"
        self.config_name = os.path.join(root_dir, "params.json")

        if os.path.isfile(self.config_name):
            with open(self.config_name, "rb") as f:
                config_data = json.load(f)
        else:
            config_data = None

        # print(config_data)

        if config_data:
            for time_str in config_data[img_type]:
                timestamp = datetime.strptime(time_str, FrameDataImage.timestamp_fmt)
                self.timestamps.append(timestamp)

        self.num_frames = 0

        while os.path.isfile(self._get_filename(self.num_frames+1)):
            self.num_frames += 1

    def get_frame_by_idx(self, idx):
        return self.read_frame(idx)

    def _get_filename(self, idx):
        return os.path.join(self.root_dir, self.base_name.format(idx+1))

    def read_frame(self, idx):
        filename = self._get_filename(idx)

        #depth_image = np.array(Image.open(filename))
        img = cv2.imread(filename, -1)

        if len(img.shape) == 2:
            img = img[:, :, None]

        return img

    def __len__(self):
        return self.num_frames


class FrameSync(object):
    def __init__(self, fd_and, fd_rgbd):
        self.fd_and = fd_and
        self.fd_rgbd = fd_rgbd

        self.sync_data = []

    def time_diff(self, idx_and, idx_rgbd):
        t_and = self.fd_and.get_timestamp(idx_and)
        t_rgbd = self.fd_rgbd.get_timestamp(idx_rgbd)

        return abs(t_and - t_rgbd), t_and > t_rgbd

    def sync(self):
        t_and_start = self.fd_and.get_timestamp(0)
        t_rgbd_start = self.fd_rgbd.get_timestamp(0)

        t_and_end = self.fd_and.get_timestamp(-1)
        t_rgbd_end = self.fd_rgbd.get_timestamp(-1)

        # android should start after rgbd
        if t_and_start < t_rgbd_start:
            idx_and_start, _ = self.fd_and.get_closest_frame(t_rgbd_start)
        else:
            idx_and_start = 0

        # android should end before rgbd
        if t_and_end > t_rgbd_end:
            idx_and_end, _ = self.fd_and.get_closest_frame(t_rgbd_end)
        else:
            idx_and_end = len(self.fd_and) - 1

        for idx_and in range(idx_and_start, idx_and_end-idx_and_start+1):
            time_and = self.fd_and.get_timestamp(idx_and)

            idx_rgbd, delta = self.fd_rgbd.get_closest_frame(time_and)

            self.sync_data.append({"idx_and": idx_and, "idx_rgbd": idx_rgbd, "delta": delta})

    def save(self, folder, transform=None, frame_limit=None):
        if not os.path.exists(os.path.join(folder, "color")):
            os.makedirs(os.path.join(folder, "color"))
        if not os.path.exists(os.path.join(folder, "depth")):
            os.makedirs(os.path.join(folder, "depth"))

        color_base = "color{}.tif"
        depth_base = "depth{}.tif"

        #limit
        # np.arange(l / nth) * nth
        # round((i/(l-1))*(n-1)) i in range[0,l]

        if frame_limit is not None:
            idx_range = np.round(
                np.arange(frame_limit) / (frame_limit - 1) * (self.sync_frame_pair_len() - 1)).astype(int)
        else:
            idx_range = range(self.sync_frame_pair_len())

        frame_config = {"frames": []}
        file_config = os.path.join(folder, "params.json")

        start_timestamp, _ = self.get_sync_frame_info(0)

        for i in idx_range:
            frame_and, frame_rgbd = self.get_sync_frame_pair(i)

            timestamp, focus = self.get_sync_frame_info(i)
            frame_config["frames"].append({"time": (timestamp - start_timestamp).total_seconds(), "focus": focus})

            if transform is not None:
                frame_and, frame_rgbd = transform(frame_and, frame_rgbd)

            file_and = os.path.join(folder, "color", color_base.format(i+1))
            file_rgbd = os.path.join(folder, "depth", depth_base.format(i+1))

            cv2.imwrite(file_and, cv2.cvtColor(frame_and, cv2.COLOR_RGB2BGR))
            cv2.imwrite(file_rgbd, frame_rgbd)

        with open(file_config, "w") as f:
            json.dump(frame_config, f)

    def _and_idx(self, sync_idx):
        return self.sync_data[sync_idx]["idx_and"]

    def _rgbd_idx(self, sync_idx):
        return self.sync_data[sync_idx]["idx_rgbd"]

    def get_sync_frame_pair(self, sync_idx):
        return [
            self.fd_and.get_frame_by_idx(self._and_idx(sync_idx)),
            self.fd_rgbd.get_frame_by_idx(self._rgbd_idx(sync_idx)),

        ]

    def get_sync_frame_info(self, sync_idx):
        return self.fd_and.get_frame_info_by_idx(self._and_idx(sync_idx))

    def sync_frame_pair_len(self):
        return len(self.sync_data)

    def __len__(self):
        return self.sync_frame_pair_len()

    def __repr__(self):
        text = ""

        for frame_info in self.sync_data:
            text += "And: {}, RGBD: {}, Delta: {}s\n".format(
                frame_info["idx_and"],
                frame_info["idx_rgbd"],
                frame_info["delta"]
            )

        max_delta = max(self.sync_data, key=lambda d: d["delta"])["delta"]

        text += "Max delta: {}s\n".format(max_delta)

        return text
