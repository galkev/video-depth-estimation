import os
import glob
import PIL
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import re
import numpy as np
import cv2
import shutil

from data.data_transforms import crop_to_aspect
from dataset_tools.rgbd_record import RgbdView
from tools.tools import load_exr, type_adv, transpose_flat_list


class ImageGridBuilder:
    final_test_directory = "/home/kevin/Documents/master-thesis/logs/final/tests"

    def __init__(self, pad=0, pad_value=1):
        self.pad = pad
        self.pad_value = pad_value
        self.directory = "/home/kevin/Documents/master-thesis/thesis/figures/pictures/compressed"
        self.test_directory = "/home/kevin/Documents/master-thesis/thesis/figures/grids"
        self.img_format = "png"

        self.rgbd_view = RgbdView()

    @staticmethod
    def natural_sorted(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    @staticmethod
    def get_filenames(path_glob):
        return ImageGridBuilder.natural_sorted(
            glob.glob(path_glob))

    @staticmethod
    def _read_vid_frame(filename, idx=0, downscale=1):
        cap = cv2.VideoCapture(filename)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print("Error cap.read")
            frame = None
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transforms.ToTensor()(
                Image.fromarray(frame).resize((frame.shape[1] // downscale,
                                               frame.shape[0] // downscale), resample=Image.ANTIALIAS))

        cap.release()

        return frame

    def _read_depth_image(self, filename):
        if filename.endswith(".exr"):
            img16 = (load_exr(filename, ["R"]) * 0xffff).astype(np.uint16)
        else:
            img16 = cv2.imread(filename, -1)[:, :, None]

        img = torch.tensor(cv2.cvtColor(self.rgbd_view.process_depth(
               img16), cv2.COLOR_BGR2RGB).astype(float) / 0xffff, dtype=torch.float).permute(2, 0, 1)

        return img

    @staticmethod
    def get_image(filename):
        return transforms.ToTensor()(Image.open(filename))

    @staticmethod
    def get_images(filenames):
        return [transforms.ToTensor()(Image.open(filename)) for filename in filenames]

    @staticmethod
    def get_image_range(root_dir, filenames, indices):
        if isinstance(indices, int):
            indices = range(indices)

        imgs = []

        for filename in filenames:
            imgs += ImageGridBuilder.get_images([os.path.join(root_dir, filename.format(i)) for i in indices])

        return imgs

    @staticmethod
    def get_test_images(tests, seqs, output_type, fixed_aspect_from_seq=None):
        img_format = "jpg"
        imgs = []

        for test_name, tensor_ids in tests:
            root_dir = glob.glob(os.path.join(ImageGridBuilder.final_test_directory, test_name + "_*", "img"))

            print(root_dir)
            assert len(root_dir) == 1

            root_dir = root_dir[0]

            for tensor_id in tensor_ids:
                for seq in seqs:
                    filename = f"seq{seq:03d}_{output_type}_tensor{tensor_id}.{img_format}"
                    imgs.append(ImageGridBuilder.get_image(os.path.join(root_dir, filename)))

        if fixed_aspect_from_seq is not None:
            target_img = imgs[0][fixed_aspect_from_seq]
            img_size = target_img.shape[-1], target_img.shape[-2]

            imgs = [crop_to_aspect(img, img_size) for img in imgs]

        return imgs

    def get_depth_images(self, filenames):
        return [
            self._read_depth_image(filename)
            for filename in filenames
        ]

    @staticmethod
    def get_images_from_video(filenames, idx=0, downscale=1):
        return [ImageGridBuilder._read_vid_frame(file, idx=idx, downscale=downscale) for file in filenames]

    def _save_grid(self, name, images, nrow):
        torchvision.utils.save_image(images, os.path.join(self.directory, name + "." + self.img_format),
                                     nrow=nrow, padding=self.pad, pad_value=self.pad_value)

    def save(self, name, images, nrow):
        assert len(images) > 0

        #if len(images) % nrow > 0:
        #    images += [torch.full_like(images[0], self.pad_value)] * (nrow - (len(images) % nrow))

        self._save_grid(name, images, nrow)

    @staticmethod
    def _make_tex_tab(num_col, content):
        # num_row = len(content) // num_col

        tex = ""

        for i in range(len(content)):
            tex += content[i] + " "

            if i != len(content):
                if (i + 1) % num_col == 0:
                    tex += r"\\"
                else:
                    tex += "&"

            tex += "\n"

        return "\\begin{tabular}{" + "c" * num_col + "}\n" + tex + "\\end{tabular}"

    @staticmethod
    def _make_tex_subfloat(caption, width, file):
        return r"\subfloat[" + caption + r"]{\includegraphics[width=" + str(width) + r"\textwidth]{" + file + r"}}"

    # https://tex.stackexchange.com/questions/239715/add-titles-for-rows-and-columns-in-a-subfloat

    def _save_image_cells(self, name, images, nrow):
        directory = os.path.join(self.test_directory, name)

        if os.path.isdir(directory):
            shutil.rmtree(directory)

        os.mkdir(directory)

        ext = "png"

        files = [f"{i // nrow}_{i % nrow}.{ext}" for i in range(len(images))]

        for img, file in zip(images, files):
            torchvision.utils.save_image(img, os.path.join(directory, file))

        return directory, files

    def save_test_latex(self, save_name, tests, seqs, row_names, img_width=None, fixed_aspect_from_seq=None,
                        pad=(0, 0.2), nrow=None, transpose=False, output_type="depth"):
        num_seqs = len(seqs)

        imgs = ImageGridBuilder.get_test_images(
            tests=tests,
            seqs=seqs,
            output_type=output_type,
            fixed_aspect_from_seq=fixed_aspect_from_seq
        )

        # col_names = [f"Image {i + 1}" for i in range(num_seqs)]
        col_names = None

        nrow = num_seqs if nrow is None else nrow

        if transpose:
            imgs = transpose_flat_list(imgs, nrow)

        self.save_latex(save_name, imgs, row_names, col_names,
                        nrow=nrow, img_width=img_width,
                        pad=pad
                        )

    def save_latex(self, name, images, row_names, col_names, nrow, img_width=None, pad=(0, 0.2)):
        num_rows = len(images) // nrow
        num_cols = len(images) // num_rows

        assert len(images) == num_rows * num_cols
        assert row_names is None or len(row_names) == num_rows
        assert col_names is None or len(col_names) == num_cols

        hpad = 0.03 + pad[0]
        vpad = -0.09 + pad[1]

        directory, files = self._save_image_cells(name, images, nrow)

        predefined_widths = {
            1: 1,
            2: 0.46,
            3: 0.3,
            4: 0.23,
            5: 0.18,
            6: 0.151
        }

        width = img_width if img_width is not None else predefined_widths[nrow]

        if col_names is not None:
            columm_tex = "& " + " & ".join([rf"\columnname{{{c}}}" for c in col_names]) + f" \\\\[{vpad}cm]" + f"\n"
        else:
            columm_tex = ""

        row_tex = ""

        for r in range(num_rows):
            row_tex += f"\\rowname{{{row_names[r]}}} &\n"

            for c in range(nrow):
                i = r * nrow + c
                row_tex += f"\\includegraphics[width={width}\\linewidth]{{{files[i]}}}"

                if c != nrow - 1:
                    row_tex += " &\n"
                elif not (r == num_rows - 1 and c == nrow - 1):
                    row_tex += f"\\\\[{vpad}cm]\n"

        width_pad = width + 0.0

        table_fmt = "c" * (nrow + 1)

        tex_table = \
            f"\\centering\n" \
            f"\\settoheight{{\\tempdima}}{{\\includegraphics[width={width_pad}\\linewidth]{{{files[0]}}}}}%\n" \
            f"{{\\setlength{{\\tabcolsep}}{{{hpad}cm}}\n" \
            f"\\begin{{tabular}}{{{table_fmt}}}\n" \
            f"{columm_tex}" \
            f"{row_tex}\n" \
            f"\\end{{tabular}}}}"

        with open(os.path.join(directory, "grid.tex"), "w") as f:
            f.write(tex_table)
