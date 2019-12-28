import cv2
import sys
import os
import glob
import numpy as np

# noinspection PyUnresolvedReferences
import pathmagic

from tools import load_blender_flow_exr
from tools.vis_tools import flow_to_vis


def main():
    # file = sys.argv[1]
    # file = "/home/kevin/Documents/master-thesis/render/s7_test/test/seq0/flow0000.exr"
    file = "/home/kevin/Downloads/seq0/flow0000.exr"
    folder = os.path.dirname(file)
    all_files = sorted(glob.glob(os.path.join(folder, "*.exr")))
    idx = all_files.index(file)

    backward = False

    flow = load_blender_flow_exr(all_files[idx], backward)

    while True:
        key = cv2.waitKey(1) & 0xff

        if key == ord('c'):
            exit(0)
        else:
            if key == ord("q") or key == ord("w"):
                if key == ord("q"):
                    idx -= 1
                else:
                    idx += 1

                idx %= len(all_files)
                flow = load_blender_flow_exr(all_files[idx], backward)
                print(all_files[idx], "Flow max", np.max(np.abs(flow)))

            img = cv2.cvtColor(flow_to_vis(flow), cv2.COLOR_RGB2BGR)
            cv2.imshow("Flow", img)

if __name__ == "__main__":
    main()
