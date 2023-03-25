import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

T_ITEM = Tuple[int, str, Tensor, Tensor, Tensor, Tensor]


class Cloth3d(Dataset):
    def __init__(self, ids_file: Path, root: Path, category: str) -> None:
        super().__init__()

        self.root = root

        with open(ids_file, "rb") as f:
            self.ids = sorted(pickle.load(f)[category], key=lambda x: int(x))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> T_ITEM:
        item_id = self.ids[index]
        npz = np.load(self.root / f"{item_id}.npz")
        pcd = torch.from_numpy(npz["pcd"])
        coords = torch.from_numpy(npz["coords"])
        labels = torch.from_numpy(npz["labels"])
        gradients = torch.from_numpy(npz["gradients"])

        return index, item_id, pcd, coords, labels, gradients

    def get_mesh(self, index: int) -> Tuple[Tensor, Tensor]:
        npz = np.load(self.root / f"{self.ids[index]}.npz")
        v = torch.from_numpy(npz["vertices"])
        t = torch.from_numpy(npz["triangles"])

        return v, t
