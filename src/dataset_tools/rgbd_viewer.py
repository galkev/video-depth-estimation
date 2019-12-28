import argparse

# noinspection PyUnresolvedReferences
import pathmagic
from dataset_tools.rgbd_recorder_window import RgbdRecorderWindow, RgbdRecorderWindowAndCtrl


def main():
    parser = argparse.ArgumentParser()

    #device = "f200"  # also for sr300 and r200
    #device = "asus"
    device = "d415"

    parser.add_argument("--dev", default=device)
    args = parser.parse_args()

    and_ctrl = False

    if and_ctrl:
        rec_wnd = RgbdRecorderWindowAndCtrl(args.dev, record_color=True, record_depth=True, sync_time_on_record=True)
    else:
        rec_wnd = RgbdRecorderWindow(args.dev, record_color=True, record_depth=True)

    # rec_wnd.view_regist = True

    rec_wnd.run()


if __name__ == "__main__":
    main()
