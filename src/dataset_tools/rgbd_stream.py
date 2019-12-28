import time

from openni import openni2
import numpy as np
import json
import cv2
from datetime import datetime


class RgbdStreamBase(object):
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def get_color(self):
        raise NotImplementedError

    def get_depth(self):
        raise NotImplementedError

    def get_color_timestamp(self):
        raise NotImplementedError

    def get_depth_timestamp(self):
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RealSenseStream(RgbdStreamBase):
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07"]

    def __init__(self, device, color_stream_req=True, depth_stream_req=True):
        self.color_stream_req = color_stream_req
        self.depth_stream_req = depth_stream_req
        self.device = device

        self.pipeline = None

        self.color_frame = None
        self.depth_frame = None

        self.color_timestamp = None
        self.depth_timestamp = None

        self.depth_scale = None

        self.align = None

        self.advanced_mode = None

    def load_preset(self, json_path="realsense_presets/MyPreset.json"):
        with open(json_path) as file:
            as_json_object = json.load(file)
        json_str = str(as_json_object).replace("'", '\"')
        self.advanced_mode.load_json(json_str)

    def start(self):
        import pyrealsense2 as rs

        config = rs.config()

        w, h = 1280, 720

        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.infrared)

        self.pipeline = rs.pipeline()

        pipeline_profile = self.pipeline.start(config)
        device = pipeline_profile.get_device()
        self.advanced_mode = rs.rs400_advanced_mode(device)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()

        self.depth_scale = depth_sensor.get_depth_scale()
        #print(self.depth_scale)

        self.align = rs.align(rs.stream.color)

        # load config after first frame arrives necessary
        self.pipeline.wait_for_frames()
        self.load_preset()

    def stop(self):
        self.pipeline.stop()

    @staticmethod
    def _frame_to_np(frame):
        depth_data = frame.as_frame().get_data()
        return np.asanyarray(depth_data)

    def read(self):
        frames = self.pipeline.wait_for_frames()

        ts_sec = frames.get_timestamp() / 1000
        self.color_timestamp = datetime.fromtimestamp(ts_sec)
        self.depth_timestamp = datetime.fromtimestamp(ts_sec)

        # print(datetime.fromtimestamp(ts_sec).strftime('%Y-%m-%d %H:%M:%S.%f'))

        frames = self.align.process(frames)

        if self.color_stream_req:
            self.color_frame = RealSenseStream._frame_to_np(frames.get_color_frame())
            #self.color_frame = RealSenseStream._frame_to_np(frames.get_infrared_frame(0))

        if self.depth_stream_req:
            self.depth_frame = RealSenseStream._frame_to_np(frames.get_depth_frame())[:, :, None]
            #self.depth_frame = RealSenseStream._frame_to_np(frames.get_infrared_frame())[:, :, None] * 255 * 255

        #print(np.min(self.depth_frame))

    def get_color(self):
        return self.color_frame

    def get_depth(self):
        return self.depth_frame

    def get_color_timestamp(self):
        return self.color_timestamp

    def get_depth_timestamp(self):
        return self.depth_timestamp


class RealSense1Stream(RgbdStreamBase):
    def __init__(self, device, color_stream_req=True, depth_stream_req=True):
        self.color_stream_req = color_stream_req
        self.depth_stream_req = depth_stream_req
        self.device = device

        self.serv = None
        self.dev = None

        self.color_frame = None
        self.depth_frame = None

        self.color_timestamp = None
        self.depth_timestamp = None

        self.use_cad = False  # color aligned to depth
        self.use_dac = True  # depth aligned to color

    def start(self):
        import pyrealsense as rs1

        self.serv = rs1.Service().__enter__()

        if self.device == "f200":
            self._setup_f200()
        else:
            print("Invalid device")

    def _setup_f200(self):
        import pyrealsense as rs1
        from pyrealsense.constants import rs_option as rs1_option

        streams = []

        if self.use_cad or self.use_dac:
            streams.append(rs1.stream.ColorStream(color_format="rgb"))
            streams.append(rs1.stream.DepthStream())

            if self.use_cad:
                streams.append(rs1.stream.CADStream(color_format="rgb"))
            if self.use_dac:
                streams.append(rs1.stream.DACStream())
        else:
            if self.color_stream_req:
                streams.append(rs1.stream.ColorStream(color_format="rgb"))
            if self.depth_stream_req:
                streams.append(rs1.stream.DepthStream())

        self.dev = self.serv.Device(streams=streams).__enter__()

        # https://github.com/IntelRealSense/hand_tracking_samples/blob/master/third_party/librealsense/src/f200.cpp
        #   Hardcoded extension controls
        #                                     option                         min  max    step     def
        #                                     ------                         ---  ---    ----     ---
        # info.options.push_back({ RS_OPTION_F200_LASER_POWER,                0,  16,     1,      16  });
        # info.options.push_back({ RS_OPTION_F200_ACCURACY,                   1,  3,      1,      2   });
        # info.options.push_back({ RS_OPTION_F200_MOTION_RANGE,               0,  100,    1,      0   });
        # info.options.push_back({ RS_OPTION_F200_FILTER_OPTION,              0,  7,      1,      5   });
        # info.options.push_back({ RS_OPTION_F200_CONFIDENCE_THRESHOLD,       0,  15,     1,      6   });
        # info.options.push_back({ RS_OPTION_F200_DYNAMIC_FPS,                0,  1000,   1,      60  });
        self._set_dev_options([
            (rs1_option.RS_OPTION_F200_LASER_POWER, 16.0),
            (rs1_option.RS_OPTION_F200_ACCURACY, 2.0),
            (rs1_option.RS_OPTION_F200_MOTION_RANGE, 0.0),
            (rs1_option.RS_OPTION_F200_FILTER_OPTION, 5.0),
            (rs1_option.RS_OPTION_F200_CONFIDENCE_THRESHOLD, 6.0),
            # (rs1_option.RS_OPTION_F200_DYNAMIC_FPS, 60.0),
        ])

    def _set_dev_options(self, options):
        import pyrealsense as rs1

        try:
            self.dev.set_device_options(*zip(*options))
        except rs1.RealsenseError as err:
            print("Failed setting options")
            print(err)

    def stop(self):
        self.dev.__exit__()
        self.serv.__exit__()

    def read(self):
        self.dev.wait_for_frames()

        self.depth_timestamp = datetime.now()
        self.color_timestamp = self.depth_timestamp

        if self.color_stream_req:
            if not self.use_cad:
                self.color_frame = self.dev.color
            else:
                self.color_frame = self.dev.cad

        if self.depth_stream_req:
            if not self.use_dac:
                self.depth_frame = self.dev.depth[:, :, None]
            else:
                self.depth_frame = self.dev.dac[:, :, None]

    def get_color(self):
        return self.color_frame

    def get_depth(self):
        return self.depth_frame

    def get_color_timestamp(self):
        return self.color_timestamp

    def get_depth_timestamp(self):
        return self.depth_timestamp


class OpenNiStream(RgbdStreamBase):
    openni_path = "/usr/local/lib"

    def __init__(self, device, color_stream_req=True, depth_stream_req=True):
        openni2.initialize(self.openni_path)
        self.device = openni2.Device.open_any()

        self.color_stream_req = color_stream_req
        self.depth_stream_req = depth_stream_req

        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30

        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30

        self.use_depth_registration = True
        self.mirror = False

        self.color_stream = None
        self.depth_stream = None

        self.color_frame = None
        self.depth_frame = None

        self.timestamp_color = None
        self.timestamp_depth = None

    def start(self):
        if self.color_stream_req:
            self._start_color()
        if self.depth_stream_req:
            self._start_depth()

    def stop(self):
        self._stop_color()
        self._stop_depth()
        openni2.unload()

    def read(self):
        if self.depth_stream_req:
            self._read_depth()
        if self.color_stream_req:
            self._read_color()

    def get_color(self):
        return self.color_frame

    def get_depth(self):
        return self.depth_frame

    def get_color_timestamp(self):
        return self.timestamp_color

    def get_depth_timestamp(self):
        return self.timestamp_depth

    def _start_color(self):
        self.color_stream = self.device.create_color_stream()
        self.color_stream.start()
        self.color_stream.configure_mode(self.color_width, self.color_height, self.color_fps,
                                         openni2.PIXEL_FORMAT_RGB888)
        self.color_stream.set_mirroring_enabled(self.mirror)

    def _start_depth(self):
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()
        self.depth_stream.configure_mode(self.depth_width, self.depth_height, self.depth_fps,
                                         openni2.PIXEL_FORMAT_DEPTH_1_MM)
        self.depth_stream.set_mirroring_enabled(self.mirror)

        self.device.set_image_registration_mode(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR if self.use_depth_registration else
            openni2.IMAGE_REGISTRATION_OFF
        )

    def _stop_color(self):
        if self.color_stream is not None:
            self.color_stream.stop()
            self.color_stream = None

    def _stop_depth(self):
        if self.depth_stream is not None:
            self.depth_stream.stop()
            self.depth_stream = None

    def _read_color(self):
        frame = self.color_stream.read_frame()
        self.timestamp_color = datetime.now()
        frame_data = frame.get_buffer_as_uint8()
        img = np.frombuffer(frame_data, dtype=np.uint8)
        img = img.reshape(self.color_height, self.color_width, 3)

        self.color_frame = img

    def _read_depth(self):
        frame = self.depth_stream.read_frame()
        #print(frame.timestamp)
        self.timestamp_depth = datetime.now()
        frame_data = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img = img.reshape(self.depth_height, self.depth_width, 1)

        self.depth_frame = img
