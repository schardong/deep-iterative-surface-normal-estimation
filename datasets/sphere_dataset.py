# coding: utf-8

import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array


class SphereDataset(InMemoryDataset):
    r"""
    Parameters
    ----------
    root : string
        Root directory where the dataset is saved.
    split : string
        Whether to load the train, validation or test dataset.
        Should be one of: 'train', 'val', or 'test'.

    Raises
    ------
    AttributeError
        If an invalid split is passed
    """
    category_files = {
        "train": "train_nonoise.txt",
        "val": "val_nonoise.txt",
        "test": "test_nonoise.txt",
    }

    def __init__(self, root, split, transform=None, pre_transform=None,
                 pre_filter=None):
        split_toks = ["train", "val", "test"]
        if not [k for k in split_toks if k in split]:
            raise AttributeError(f"Invalid split (\"{split}\")")

        self.split = split
        self.category = "nonoise"

        super(SphereDataset, self).__init__(root, None, None, None)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f"{self.split}_{self.category}.txt"

    @property
    def processed_file_names(self):
        return f"{self.split}_{self.category}.pt"

    def process(self):
        path_file = self.raw_paths
        with open(path_file[0], "r") as f:
            filenames = f.read().split('\n')[:-1]

        data_list = []
        for fname in filenames:
            pos = read_txt_array(osp.join(self.raw_dir, fname + ".xyz"))
            normals = read_txt_array(osp.join(self.raw_dir, fname + ".normals"))
            sdf = read_txt_array(osp.join(self.raw_dir, fname + ".sdf"))

            data = Data(pos=pos,
                        x=torch.cat([pos, normals], dim=1),
                        y=sdf)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
