import torch
import numpy as np
import pandas as pd
import torch.utils.data as data


class FeatureDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.classes, self.counts = np.unique(self.labels, return_counts=True)


    def __getitem__(self, index):
        feats = self.features[index, :]
        labels = self.labels[index]

        return feats, labels, index

    def __len__(self):
        return len(self.labels)


def generate_dataloader(features, labels, batch_size, sf_tag, dp_tag):
    
    dataset = FeatureDataset(features=features, labels=labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=sf_tag, drop_last=dp_tag)

    return loader