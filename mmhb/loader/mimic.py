"""
MIMIC dataloader adapted for MMHB base classes
"""

from mmhb.loader import MMDataset
from mmhb.utils import Config
from pathlib import Path
from typing import List
import pickle
import numpy as np
import torch
from typing import Union


class MimicDataset(MMDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        expand: bool = False,
        modalities: List = ["tab", "ts"],
        concat: bool = False,
        task: int = 0,
        **kwargs
    ):
        super().__init__(
            data_path=data_path, expand=expand, modalities=modalities, **kwargs
        )
        self.concat = concat
        self.expand = expand
        f = open(data_path, "rb")
        datafile = pickle.load(f)
        f.close()
        X_t = datafile["ep_tdata"]
        X_s = datafile["adm_features_all"]

        X_t[np.isinf(X_t)] = 0
        X_t[np.isnan(X_t)] = 0
        X_s[np.isinf(X_s)] = 0
        X_s[np.isnan(X_s)] = 0

        X_s_avg = np.average(X_s, axis=0)
        X_s_std = np.std(X_s, axis=0)
        X_t_avg = np.average(X_t, axis=(0, 1))
        X_t_std = np.std(X_t, axis=(0, 1))

        # normalise data
        for i in range(len(X_s)):
            X_s[i] = (X_s[i] - X_s_avg) / X_s_std
            for j in range(len(X_t[0])):
                X_t[i][j] = (X_t[i][j] - X_t_avg) / X_t_std

        static_dim = len(X_s[0])
        timestep = len(X_t[0])
        series_dim = len(X_t[0][0])
        if concat:
            # flatten ts if we want to concatenate later
            X_t = X_t.reshape(len(X_t), timestep * series_dim)
        if task < 0:
            y = datafile["adm_labels_all"][:, 1]
            admlbl = datafile["adm_labels_all"]
            le = len(y)
            for i in range(0, le):
                if admlbl[i][1] > 0:
                    y[i] = 1
                elif admlbl[i][2] > 0:
                    y[i] = 2
                elif admlbl[i][3] > 0:
                    y[i] = 3
                elif admlbl[i][4] > 0:
                    y[i] = 4
                elif admlbl[i][5] > 0:
                    y[i] = 5
                else:
                    y[i] = 0
        else:
            y = datafile["y_icd9"][:, task]
            le = len(y)

        self.targets = torch.Tensor(y).long()
        self.X_t = torch.Tensor(X_t)
        self.X_s = torch.Tensor(X_s)

    def __getitem__(self, idx):
        tensors = []
        if "tab" in self.modalities:
            tensor = self.X_t[idx]
            if self.expand:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)

        if "ts" in self.modalities:
            tensors.append(self.X_s[idx])

        return tensors, self.targets[idx]


if __name__ == "__main__":
    config = Config("../mm-lego/config/config_dev.yml").read()
    mimic = MimicDataset(**config.data.mimic.to_dict())
    print(mimic[0])
    print(mimic.targets)
    print(mimic.targets.unique())
