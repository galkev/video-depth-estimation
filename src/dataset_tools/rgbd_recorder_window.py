import numpy as np
import cv2

# noinspection PyUnresolvedReferences
import pathmagic
from dataset_tools.rgbd_stream import OpenNiStream, RealSenseStream, RealSense1Stream
from dataset_tools.rgbd_record import RgbdView, RgbdRecorderImage
from dataset_tools.fps_counter import FpsCounter
from dataset_tools.simple_socket import SimpleSocket
from scripts.helper.send_time_to_dev import sync_dev_time


class RgbdRecorderWindow(object):
    def __init__(self, device, record_color=False, record_depth=True):
        self.device = device
        self.record_color = record_color
        self.record_depth = record_depth

        self.view_regist = False
        self.no_processing_on_record = False

        # depth_cmap, depth_invalid_color = None, np.array([0x8fff, 0, 0])
        # depth_cmap, depth_invalid_color = "viridis", np.array([0, 0, 0])
        # normalize_depth = True

        # self.view = RgbdView(depth_cmap=depth_cmap, depth_invalid_color=depth_invalid_color, normalize_depth=normalize_depth)
        self.view = RgbdView()

        self.recorder = RgbdRecorderImage()

        if self.device == "asus":
            self.stream_class = OpenNiStream
        elif self.device == "f200":
            self.stream_class = RealSense1Stream
        elif self.device == "d415":
            self.stream_class = RealSenseStream
        else:
            return

    def run(self):
        fps_counter = FpsCounter()

        with self.stream_class(
                self.device,
                color_stream_req=self.record_color,
                depth_stream_req=self.record_depth) as stream:
            while True:
                fps_counter.step()

                stream.read()

                color, depth = None, None

                if self.record_color:
                    color = stream.get_color()

                if self.record_depth:
                    depth = stream.get_depth()

                view_label = str(fps_counter)

                if self.recorder.is_recording():
                    view_label += "|{:.1f}s".format(self.recorder.get_record_time())

                if self.recorder.is_recording():
                    color_bgr, _ = self.recorder.record_frame(
                        color,
                        depth,
                        timestamp_color=stream.get_color_timestamp(),
                        timestamp_depth=stream.get_depth_timestamp()
                    )

                    if self.no_processing_on_record:
                        RgbdView.cv_show(color_bgr, depth, label=view_label)
                    else:
                        self.view.show(color, depth, label=view_label)
                else:
                    if self.view_regist:
                        self.view.show_registration(color, depth)
                    else:
                        self.view.show(color, depth, label=view_label)

                key = cv2.waitKey(1) & 0xff
                if key == ord('r'):
                    if not self.recorder.is_recording():
                        self.on_record_start()
                        self.recorder.start_record(self.record_color, self.record_depth)
                        print("Start record")
                    else:
                        self.on_record_end()
                        self.recorder.end_record()
                        print("End record")
                elif key == ord('p'):
                    if self.view_regist:
                        self.recorder.save_frame(color=self.view.get_registration_view_inst(color, depth))
                    else:
                        self.recorder.save_frame(color=color, depth=depth, depth_view=self.view)
                    print("Saved Frame")
                elif key == ord('u'):
                    stream.load_preset()
                elif key == ord('q'):
                    break

    def on_record_start(self):
        pass

    def on_record_end(self):
        pass


class RgbdRecorderWindowAndCtrl(RgbdRecorderWindow):
    def __init__(self, device, record_color=False, record_depth=True, sync_time_on_record=True):
        super().__init__(device, record_color, record_depth)

        self.sock = SimpleSocket()
        self.sync_time_on_record = sync_time_on_record

    def send_message(self, cmd, *args):
        if self.sock.send(cmd, *args).decode() != "ok":
            print("Failure")

    def on_record_start(self):
        self.sock.connect()
        if self.sync_time_on_record:
            sync_dev_time(self.sock)

        self.send_message("start_rec")
        self.sock.close()

    def on_record_end(self):
        self.sock.connect()
        self.send_message("stop_rec")
        self.sock.close()
