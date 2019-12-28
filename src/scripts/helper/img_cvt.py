import cv2
import os
import numpy as np
import glob as glob


def s2linbak(x):
    a = 0.055

    res = np.ones_like(x)
    mask1 = x <= 0.04045
    mask2 = (0.04045 < x) & (x < 1.0)

    res[mask1] = x[mask1] * (1.0 / 12.92)
    res[mask2] = pow((x[mask2] + a) * (1.0 / (1 + a)), 2.4)

    return res


def s2lin(x):
    return np.where(x >= 0.04045, ((x + 0.055) / 1.055)**2.4, x/12.92)


def color_corr_reverse(file_in, file_out):
    img = cv2.imread(file_in, -1)
    img = (s2lin(img / 0xffff) * 0xffff).astype(np.uint16)
    cv2.imwrite(file_out, img)


def gamma_calc():
    img1 = cv2.imread("/home/kevin/Documents/master-thesis/render/cmp/1.tif")
    img2 = cv2.imread("/home/kevin/Documents/master-thesis/render/cmp/2.tif")

    gammas = np.log(img1) / np.log(img2)

    print(gammas)


def main():
    file_wildcast = "/storage/slurm/galim/Documents/master-thesis/datasets/dining_room/data/*/depth*"

    files = glob.glob(file_wildcast)

    for i, file in enumerate(files, 1):
        out_file = file.replace("/dining_room/", "/dining_room_cvt/")
        out_dir = os.path.dirname(out_file)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        color_corr_reverse(file, out_file)

        print(i, "/", len(files))


if __name__ == "__main__":
    main()
