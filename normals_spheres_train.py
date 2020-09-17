#!/usr/bin/env python3
# coding: utf-8


import torch
from torch_geometric.data import DataLoader
from datasets.sphere_dataset import SphereDataset


args = {
    "model_name": "sphere_net_epoch_{}.pt",
    "dataset_path": "data/spheres",
    "k_train": 16,
    "iterations": 4,
}


# test_dataset = SphereDataset(args["dataset_path"], "test")

train_dataset = SphereDataset(args["dataset_path"], "train")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                          pin_memory=True, num_workers=4)

val_dataset = SphereDataset(args["dataset_path"], "val")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


class SDFEstimation(torch.nn.Module):
    def __init__(self):
        super(SDFEstimation, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, old_weights, pos, batch, sdf, )
