import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict

import torch
from hesiod import get_out_dir, hcfg, hmain
from torch import Tensor
from torch.utils.data import DataLoader

from data.cloth3d import Cloth3d
from meshudf.meshudf import get_mesh_from_udf
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import get_o3d_mesh_from_tensors, progress_bar, random_point_sampling

import open3d as o3d  # isort: skip

if len(sys.argv) != 2:
    print("Usage: python export_meshes.py <run_cfg_file>")
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

    latent_size = hcfg("latent_size", int)
    num_points_pcd = hcfg("num_points_pcd", int)
    udf_max_dist = hcfg("udf_max_dist", float)

    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    coords_encoder = CoordsEncoder()

    decoder_cfg = hcfg("decoder", Dict[str, Any])
    decoder = CbnDecoder(
        coords_encoder.out_dim,
        latent_size,
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers"],
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    dset_test_ids_file = Path(hcfg("dset.test_ids_file", str))
    dset_category = hcfg("dset.category", str)
    dset_root = Path(hcfg("dset.root", str))

    bs = hcfg("val_bs", int)

    test_dset = Cloth3d(dset_test_ids_file, dset_root, dset_category)
    test_loader = DataLoader(test_dset, bs, num_workers=8)

    for batch in progress_bar(test_loader, "Exporting"):
        _, item_ids, pcds, _, _, _ = batch
        bs = pcds.shape[0]
        pcds = pcds.cuda()

        pcds = random_point_sampling(pcds, num_points_pcd)

        with torch.no_grad():
            latent_codes = encoder(pcds)

        for i in progress_bar(range(bs), "Meshing"):
            lat = latent_codes[i].unsqueeze(0)

            def udf_func(c: Tensor) -> Tensor:
                c = coords_encoder.encode(c.unsqueeze(0))
                p = decoder(c, lat).squeeze(0)
                p = torch.sigmoid(p)
                p = (1 - p) * udf_max_dist
                return p

            v, t = get_mesh_from_udf(
                udf_func,
                coords_range=(-1, 1),
                max_dist=udf_max_dist,
                N=512,
                max_batch=2**16,
                differentiable=False,
            )

            pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
            mesh_path = get_out_dir() / f"meshes_test/{item_ids[i]}.obj"
            mesh_path.parent.mkdir(exist_ok=True, parents=True)
            o3d.io.write_triangle_mesh(str(mesh_path), pred_mesh_o3d)


if __name__ == "__main__":
    main()
