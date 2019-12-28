import csv
import json
import itertools
import collections
import os
import torch
import pickle
import OpenEXR
import Imath
import numpy as np
import random
import re
from PIL import Image
import torchvision
import string
import unicodedata


def transpose_flat_list(x, n):
    x_t = []

    m = len(x) // n
    assert len(x) == m * n

    for i in range(n):
        for j in range(m):
            x_t.append(x[j * n + i])

    return x_t


def transpose_lists_2d(l):
    return list(map(list, zip(*l)))


def flat_zip(list1, list2):
    return sum([list(x) for x in zip(list1, list2)], [])


def valid_filename(filename):
    validFilenameChars = "-_. %s%s" % (string.ascii_letters, string.digits)

    cleanedFilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
    cleanedFilename = ''.join(chr(c) for c in cleanedFilename if chr(c) in validFilenameChars)
    cleanedFilename = cleanedFilename.replace(" ", "_")

    return cleanedFilename


def get_free_filename(path):
    filename = None

    for i in range(9999999):
        filename = path.format(i)

        if not os.path.isfile(filename):
            break

    return filename


def dump_tensor(tensor, normalize=False):
    dir = "/home/kevin/Documents/master-thesis/logs/dump"

    filename = get_free_filename(os.path.join(dir, "tensor{}.tif"))

    if normalize:
        tensor = normalize_tensor(tensor)

    torchvision.utils.save_image(tensor, filename)


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()

    return (tensor - min_val) / (max_val - min_val)


def is_size_equal(size1, size2):
    return all((s1 == s2 for s1, s2 in zip(size1, size2)))


def is_size_greater(size1, size2):
    return all((s1 >= s2 for s1, s2 in zip(size1, size2))) and any((s1 > s2 for s1, s2 in zip(size1, size2)))


def is_size_less(size1, size2):
    return all((s1 <= s2 for s1, s2 in zip(size1, size2))) and any((s1 < s2 for s1, s2 in zip(size1, size2)))


def torch_expand_back_as(x, target):
    if isinstance(x, torch.Tensor):
        for _ in range(len(target.shape) - len(x.shape)):
            x = x.unsqueeze(-1)

    return x


class GetItemBase:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.get_indices(range(*key.indices(len(self))))
        elif isinstance(key, list):
            return self.get_indices(key)
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)

            return self.get(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def get_indices(self, keys):
        return [self.get(i) for i in keys]


"""
class GetItemHelper:
    def __init__(self, length, get_func, get_indices_func=None):
        self.length = length
        self.get_func = get_func
        self.get_indices_func = get_indices_func if get_indices_func is not None \
            else lambda key: [get_func(ii) for ii in key]

    def __getitem__(self, key):
        if isinstance(key, range):
            # get the start, stop, and step from the slice
            return self.get_indices_func(key)
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += self.length
            if key < 0 or key >= self.length:
                raise IndexError("The index (%d) is out of range." % key)

            return self.get_func(key)
        else:
            raise TypeError("Invalid argument type.")
"""


def _type_adv_rec(obj):
    if isinstance(obj, list):
        return [_type_adv_rec(o) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(_type_adv_rec(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: _type_adv_rec(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return "{} {}".format(str(type(obj).__name__), list(obj.size()))
    else:
        return str(type(obj))


def type_adv(obj):
    #return _type_adv_rec(obj)
    return json.dumps(_type_adv_rec(obj), indent=4)


def dict_to_dev(data, device):
    return {k: v.to(device) for k, v in data.items()}


def squeeze_dict(dic):
    return {k: (v[0] if isinstance(v, torch.Tensor) else v) for k, v in dic.items()}


def unsqueeze_dict(dic):
    return {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in dic.items()}


def to_tensor_crop(img, crop=None):
    from data.data_transforms import ToFloatTensor, CenterCrop

    tensor = ToFloatTensor()(img)

    if crop is not None:
        tensor = CenterCrop(crop)(tensor)

    return tensor


def load_tensor_image(path, crop=None):
    img = Image.open(path)
    tensor = to_tensor_crop(img, crop)

    return tensor


def load_tensor_depth(path, crop=None):
    img = load_exr(path, ["R"])
    tensor = to_tensor_crop(img, crop)

    return tensor


# alternating_range(5) -> [0, 4, 1, 3, 2]
def alternating_range(stop):
    center = stop // 2
    r = sum(([a, b] for a, b in zip(range(center), range(stop - 1, center - 1, -1))), [])

    if len(r) < stop:
        r.append(center)

    return r


def model_freeze_state_repr(model, depth=0):
    frozen = all(not p.requires_grad for p in model.parameters())
    not_frozen = all(p.requires_grad for p in model.parameters())

    if frozen:
        state = "Frozen"
    elif not_frozen:
        state = "Not frozen"
    else:
        state = "Mixed"

    text = "{} [{}]".format(model._get_name(), state)

    if not frozen and not not_frozen:
        spaces = " " * (depth + 1)

        text += "\n" + spaces + ("\n" + spaces).join(
            model_freeze_state_repr(mod, depth + 1) for mod in model.children())

    return text


class IdentityModule(torch.nn.Module):
    def __init__(self, in_channels=-1, out_channels=None, num_out=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_out = num_out

    def get_in_channels_count(self):
        return self.in_channels

    def forward(self, x):
        if self.out_channels is not None:
            x = x[:, :, :self.out_channels]

        if self.num_out is not None:
            x = tuple([x] * self.num_out)

        return x


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def deterministic(seed=412):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    print("Seeeeeeeeeeeeeding!!!")


def module_flat_str(module, module_filter=None):
    if module_filter is not None and type(module) in module_filter:
        return None
    elif len(module._modules.items()) == 0:
        return str(module)
    else:
        sub_mod = [module_flat_str(v, module_filter) for _, v in module._modules.items()]
        sub_mod_valid = filter(lambda x: x is not None, sub_mod)
        return "\n".join(sub_mod_valid)


def dict_public_items(d, include=None):
    nd = {}

    if include is None:
        include = []

    for k, v in d.items():
        if not k.startswith("_") or k in include:
            nd[k] = v

    return nd


def to_json_dict(obj):
    if _has_json_support(obj):
        return obj._to_json()
    elif isinstance(obj, torch.nn.Sequential):
        return {str(i): to_json_dict(v) for i, v in enumerate(obj)}
    elif isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in dict_public_items(obj).items()}
    else:
        return str(obj)


def obj_to_json(obj):
    return {k: to_json_dict(v) for k, v in dict_public_items(vars(obj), ["_modules"]).items()}


def json_support(cls):
    def to_json(self):
        return obj_to_json(self)

    setattr(cls, "_to_json", to_json)
    return cls


def _has_json_support(obj):
    try:
        if hasattr(obj, "_to_json"):
            return True
    except NotImplementedError:
        pass

    return False


def tensor_to_np_img(tensor):
    return tensor.permute(1, 2, 0).detach().cpu().numpy()


def np_img_to_tensor(np_img):
    return torch.from_numpy(np_img).permute(2, 0, 1)


def load_exr(file, channels):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr = OpenEXR.InputFile(file)

    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    img = np.stack([
        np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(size[1], size[0])
        for c in channels], axis=2)

    return img


def load_blender_flow_exr(file, backward=False):
    # R: pixel displacement in X from current frame to previous frame
    # G: pixel displacement in Y from current frame to previous frame
    # B: pixel displacement in X from next frame to current frame
    # A: pixel displacement in Y from next frame to current frame
    channels = ["R", "G"] if backward else ["B", "A"]
    flow = load_exr(file, channels)

    if backward:
        flow[:, :, 1] *= -1  # y axis is reversed in blender
    else:
        # flip flow to point forward
        flow[:, :, 0] *= -1

    return flow


def obj_to_str(obj):
    attr = vars(obj)
    return "{}(\n{}\n)".format(
        obj.__class__.__name__,
        "\n".join("    {}: {}".format(k, v) for k, v in attr.items())
    )


def default_device(device=None):
    if device is not None:
        return device
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device):
    if device.type == "cuda":
        return "{} {}GB".format(
            torch.cuda.get_device_name(device),
            round(torch.cuda.get_device_properties(device).total_memory / 1024**3)
        )
    else:
        return "CPU"


def dict_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def dict_str_desc(dictionary):
    return json.dumps(dictionary, indent=4)


def dict_cross_prod(dict_arr, ignore_keys=None):
    new_dict = {}

    for k, v in dict_arr.items():
        if type(v) == dict and not (ignore_keys is not None and k in ignore_keys):
            new_dict[k] = dict_cross_prod(v, ignore_keys)
        elif type(v) == list:
            new_dict[k] = v
        else:
            new_dict[k] = [v]

    return dict_product(new_dict)


def arg_cross_prod(**kwargs):
    return dict_cross_prod(kwargs)


def dict_update_rec(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update_rec(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def save_file(filename, data, create_dirs=False):
    if data is not None:
        if create_dirs:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        ext = os.path.splitext(filename)[1]

        if ext == ".pt" or ext == ".pth":
            torch.save(data.state_dict() if not isinstance(data, dict) else data, filename)
        elif ext == ".pkl":
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif ext == ".json":
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        elif ext == ".csv":
            return _save_file_csv(filename, data)
        else:
            with open(filename, "w") as f:
                f.write(data)


def load_file(filename, obj=None):
    if os.path.isfile(filename):
        ext = os.path.splitext(filename)[1]

        if ext == ".pt" or ext == ".pth":
            if obj is not None:
                obj.load_state_dict(torch.load(filename))
                return obj
            else:
                return torch.load(filename)
        elif ext == ".pkl":
            with open(filename, 'rb') as f:
                return pickle.load(f)
        elif ext == ".json":
            with open(filename, 'r') as f:
                return json.load(f)
        elif ext == ".csv":
            return _load_file_csv(filename)
        else:
            with open(filename, "r") as desc_file:
                return desc_file.read()
    else:
        raise Exception("Error load_file: File '{}' not found".format(filename))


def _save_file_csv(filename, list_data):
    with open(filename, 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        if type(list_data[0]) is list:
            wr.writerows(list_data)
        else:
            for ele in list_data:
                wr.writerow([ele])


def _load_file_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        list_data = list(reader)

    if len(list_data[0]) == 1:
        list_data = [row[0] for row in list_data]

    return list_data
