import sys

sys.path.append("..")

from pathlib import Path

import torch
from hesiod import get_out_dir, hcfg, hmain
from torch.utils.data import DataLoader

from data.cloth3d import Cloth3d
from models.dgcnn import Dgcnn
from utils import progress_bar, random_point_sampling

if len(sys.argv) != 2:
    print("Usage: python export_codes.py <run_cfg_file>")
    exit(1)


@hmain(
    base_cfg_dir="../cfg/bases",
    run_cfg_file=sys.argv[1],
    parse_cmd_line=False,
    out_dir_root="../logs",
)
def main() -> None:
    ckpt_path = list((get_out_dir() / "ckpts").glob("last*.pt"))[0]
    ckpt = torch.load(ckpt_path)

    num_points_pcd = hcfg("num_points_pcd", int)

    latent_size = hcfg("latent_size", int)
    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    dset_train_ids_file = Path(hcfg("dset.train_ids_file", str))
    dset_test_ids_file = Path(hcfg("dset.test_ids_file", str))
    dset_category = hcfg("dset.category", str)
    dset_root = Path(hcfg("dset.root", str))

    bs = hcfg("val_bs", int)

    train_dset = Cloth3d(dset_train_ids_file, dset_root, dset_category)
    train_loader = DataLoader(train_dset, bs, num_workers=8)
    test_dset = Cloth3d(dset_test_ids_file, dset_root, dset_category)
    test_loader = DataLoader(test_dset, bs, num_workers=8)

    for split, loader in [("train", train_loader), ("test", test_loader)]:
        all_latent_codes = []

        for batch in progress_bar(loader, split):
            _, _, pcds, _, _, _ = batch
            pcds = pcds.cuda()
            pcds = random_point_sampling(pcds, num_points_pcd)

            with torch.no_grad():
                latent_codes = encoder(pcds)

            all_latent_codes.append(latent_codes.detach().cpu())

        all_latent_codes = torch.cat(all_latent_codes, dim=0)
        latent_codes_path = get_out_dir() / f"latent_codes/{split}_all.pt"
        latent_codes_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(all_latent_codes, latent_codes_path)

        mean_latent_code = torch.mean(all_latent_codes, dim=0)
        torch.save(mean_latent_code, latent_codes_path.parent / f"{split}_mean.pt")


if __name__ == "__main__":
    main()
