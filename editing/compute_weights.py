import sys

sys.path.append("..")

import pickle
from pathlib import Path
from typing import List

import torch
from hesiod import hcfg, hmain
from sklearn.linear_model import LogisticRegression

from data.codecls import CodeClassification


@hmain(
    base_cfg_dir="../cfg/bases",
    template_cfg_file="../cfg/compute_weights.yaml",
    create_out_dir=False,
)
def main() -> None:
    list_file = Path(hcfg("dset.classification_file", str))
    all_codes_file = Path(hcfg("dset.train_latent_codes", str))
    train_ids_file = Path(hcfg("dset.train_ids_file", str))
    category = hcfg("dset.category", str)
    classes = hcfg("dset.classes", List[str])

    with open(train_ids_file, "rb") as f:
        all_codes_ids = sorted(pickle.load(f)[category], key=lambda x: int(x))

    dset = CodeClassification(all_codes_file, all_codes_ids, list_file, classes)

    x = []
    y = []
    for i in range(len(dset)):
        code, label = dset[i]
        x.append(code)
        y.append(label)

    x = torch.stack(x, dim=0).numpy()
    y = torch.stack(y, dim=0).numpy()

    model = LogisticRegression()
    model.fit(x, y)
    importance = model.coef_[0]

    weights = torch.tensor(importance).float()
    weights_file = Path(hcfg("weights_file", str))
    torch.save(weights, weights_file)


if __name__ == "__main__":
    main()
