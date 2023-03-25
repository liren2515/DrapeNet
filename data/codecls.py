from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CodeClassification(Dataset):
    def __init__(
        self,
        all_codes_file: Path,
        all_codes_ids: List[str],
        list_file: Path,
        classes: List[str],
    ) -> None:
        self.all_codes_ids = all_codes_ids
        self.all_codes = torch.load(all_codes_file)

        with open(list_file, "rt") as f:
            lines = [line.strip() for line in f.readlines()]

        self.codes_ids = []
        self.labels = []
        for line in lines:
            code_id, label = line.split(",")
            self.codes_ids.append(code_id)
            self.labels.append(classes.index(label))

    def __len__(self) -> int:
        return len(self.codes_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        code_id = self.codes_ids[index]
        idx = self.all_codes_ids.index(code_id)
        code = self.all_codes[idx]
        label = torch.tensor(self.labels[index]).float()

        return code, label
