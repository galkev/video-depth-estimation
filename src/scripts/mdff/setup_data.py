import os
import random
import pickle
from PIL import Image

# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir
from tools import tools
from data import data_transforms, VideoDepthFocusData


def setup_data(root_dir, out_dir, split_train_at=145, data_folder_name="mDFFDataset"):
    image_ids_dict = {
        "train": None,
        "val": None,
        "test": None
    }

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for data_type in ["train", "test"]:
        focalstack_folder = "focalstacks"
        depth_folder = "depth"
        depthmap_name = "depth.png"
        focalstack_size = 10

        data_dir = data_type

        image_ids = [f.name for f in os.scandir(
            os.path.join(root_dir, data_folder_name, data_dir, focalstack_folder)) if f.is_dir()]

        random.shuffle(image_ids)

        if data_type == "train":
            image_ids_dict["train"] = image_ids[:split_train_at]
            image_ids_dict["val"] = image_ids[split_train_at:]
        else:
            image_ids_dict["test"] = image_ids

    for data_type, image_ids in image_ids_dict.items():
        data_dir = "test" if data_type == "test" else "train"

        data = []
        val_crops_list = []

        for image_id in image_ids:
            focalstack_paths = \
                [os.path.join(data_dir, focalstack_folder, image_id, "{}.jpg".format(i))
                 for i in range(focalstack_size)]

            depth_path = \
                os.path.join(data_dir, depth_folder, image_id, depthmap_name)

            depth_img = Image.open(os.path.join(root_dir, data_folder_name, depth_path))
            to_float_tensor = data_transforms.ToFloatTensor()

            val_crops = data_transforms.RandomCrop.get_all_valid_crop_pos(
                to_float_tensor(depth_img), VideoDepthFocusData.crop_size)

            print("Found {} crops".format(len(val_crops)))

            data.append(focalstack_paths + [depth_path])
            val_crops_list.append(val_crops)

        tools.save_file(os.path.join(out_dir, "{}_data.csv".format(data_type)), data)

        crops_file = os.path.join(out_dir, "{}_data_valid_crops.pkl".format(data_type))
        with open(crops_file, "wb") as f:
            pickle.dump(val_crops_list, f)


if __name__ == "__main__":
    setup_data(proj_dir("datasets"), "mdff_setup")
