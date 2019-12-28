import os
import torch.utils.data
import h5py
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir="",
                 data_type="",
                 data_folder_name=""):
        self.data_type = data_type
        self.data_path = os.path.join(root_dir, data_folder_name)

        self.cached_data = None
        self.cached_data_transformed = False

    def __repr__(self):
        return "{}(type={}, path={})".format(
            self.__class__.__name__,
            self.data_type,
            self.data_path
        )

    def preload_data(self, preload_mode):
        if preload_mode == "transformed":
            print("Preloading data transformed")
            cached_data_transformed = True
        elif preload_mode == "untransformed":
            print("Preloading data untransformed")
            cached_data_transformed = False
        else:
            print("Not preloading data.")
            return False

        if cached_data_transformed:
            self.cached_data = [self.get_item(i) for i in range(len(self))]
        else:
            self.cached_data = [self.load_item(i) for i in range(len(self))]

        self.cached_data_transformed = cached_data_transformed

        return True

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)

            return self.get_item(key)
        else:
            raise TypeError("Invalid argument type.")

    def get_item(self, key):
        return self.transform(key, self.load_item(key))

    def load_item(self, key):
        return self.load_input(key), self.load_target(key)

    def transform(self, key, data):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def load_input(self, key):
        raise NotImplementedError

    def load_target(self, key):
        raise NotImplementedError

    def rev_transform(self, input_data, target_data, output_data):
        raise NotImplementedError

    """
    def get_transform(self, key, input_data, target_data):
        raise NotImplementedError
    """


class DatasetExpand(Dataset):
    def __init__(self,
                 dataset,
                 factor):
        super().__init__()

        self.dataset = dataset
        self.factor = factor

    def __len__(self):
        return self.factor * len(self.dataset)

    def load_input(self, key):
        return self.dataset.load_input(key // self.factor)

    def load_target(self, key):
        return self.dataset.load_target(key // self.factor)

    def load_item(self, key):
        return self.dataset.load_item(key // self.factor)

    def transform(self, key, data):
        return self.dataset.transform(key // self.factor, data)

    def rev_transform(self, input_data, target_data, output_data):
        return self.dataset.rev_transform(input_data, target_data, output_data)

    def __repr__(self):
        return "DatasetExpand({}, {})".format(
            self.dataset.__repr__(),
            self.factor
        )


class DatasetSubset(Dataset):
    def __init__(self,
                 dataset,
                 indices):
        super().__init__()

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "DatasetSubset({}, {})".format(
            self.dataset.__repr__(),
            len(self.indices)
        )

    def load_input(self, key):
        return self.dataset.load_input(self.indices[key])

    def load_target(self, key):
        return self.dataset.load_target(self.indices[key])

    def load_item(self, key):
        return self.dataset.load_item(self.indices[key])

    def transform(self, key, data):
        return self.dataset.transform(self.indices[key], data)

    def rev_transform(self, input_data, target_data, output_data):
        return self.dataset.rev_transform(input_data, target_data, output_data)


class DatasetJoin(Dataset):
    def __init__(self, datasets):
        super().__init__()

        self.datasets = datasets
        self.dataset_intervals = np.cumsum([0] + [len(d) for d in self.datasets])

        print(self.dataset_intervals, len(self))

    @property
    def depth_output_indices(self):
        return self.datasets[0].depth_output_indices

    def __len__(self):
        return self.dataset_intervals[-1]

    def __repr__(self):
        return "DatasetJoin(({}), {})".format(
            ", ".join([d.__repr__() for d in self.datasets]),
            len(self)
        )

    def _get_dataset_key(self, key):
        data_idx = list(key >= self.dataset_intervals).index(False) - 1
        rel_key = key - self.dataset_intervals[data_idx]

        # print(key, "->", data_idx, rel_key)

        return self.datasets[data_idx], rel_key

    def load_input(self, key):
        dataset, rel_key = self._get_dataset_key(key)
        return dataset.load_input(rel_key)

    def load_target(self, key):
        dataset, rel_key = self._get_dataset_key(key)
        return dataset.load_target(rel_key)

    def load_item(self, key):
        dataset, rel_key = self._get_dataset_key(key)
        return dataset.load_item(rel_key)

    def transform(self, key, data):
        dataset, rel_key = self._get_dataset_key(key)
        return dataset.transform(rel_key, data)

    def rev_transform(self, input_data, target_data, output_data):
        raise Exception("Not implemented")
        # dataset, rel_key = self._get_dataset_key(key)
        # return dataset.rev_transform(input_data, target_data, output_data)


class EmptyDataset(Dataset):
    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)

        self.length = length

    def transform(self, key, data):
        return None

    def __len__(self):
        return self.length

    def load_input(self, key):
        return None

    def load_target(self, key):
        return None

    def rev_transform(self, input_data, target_data, output_data):
        return None


class H5Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 data_type,
                 data_folder_name,
                 files,
                 keys):
        super().__init__(root_dir, data_type, data_folder_name)

        self.keys = keys

        self.data = h5py.File(os.path.join(self.data_path, files[data_type]), "r")

    def preload_data(self, preload_mode):
        if super().preload_data(preload_mode):
            self.data.close()
            return True
        else:
            return False

    # data_type: 0 -> input, 1 -> target
    def get_key(self, data_type):
        return self.keys[self.data_type][data_type]

    def __len__(self):
        return len(self.data[self.get_key(0)])

    def load_input(self, key):
        return self.data[self.get_key(0)][key]

    def load_target(self, key):
        return self.data[self.get_key(1)][key]

    def get_transform(self, key, input_data, target_data):
        raise NotImplementedError

    def rev_transform(self, input_data, target_data, output_data):
        raise NotImplementedError
