import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict

import torch
from hesiod import get_out_dir, hcfg, hmain
from torch import Tensor

from data.cloth3d import Cloth3d
from meshudf.meshudf import get_mesh_from_udf
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import get_o3d_mesh_from_tensors, progress_bar, random_point_sampling

import open3d as o3d  # isort: skip


@hmain(
    base_cfg_dir="../cfg/bases",
    template_cfg_file="../cfg/edit.yaml",
    out_dir_root="../logs",
)
def main() -> None:
    ckpt_path = Path(hcfg("encdec_ckpt_path", str))
    ckpt = torch.load(ckpt_path)

    num_points_pcd = hcfg("num_points_pcd", int)
    latent_size = hcfg("latent_size", int)
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
    test_dset = Cloth3d(dset_test_ids_file, dset_root, dset_category)

    num_steps = hcfg("num_steps", int)
    step_size = hcfg("step_size", float)
    required_garment_id = hcfg("required_garment_id", str)

    weights_file = Path(hcfg("dset.weights_file", str))
    w = torch.load(weights_file)
    abs_min = torch.min(torch.abs(w))
    abs_max = torch.max(torch.abs(w))
    normalized_w = (torch.abs(w) - abs_min) / (abs_max - abs_min)
    normalized_w[normalized_w < 0.5] = 0
    mask = normalized_w > 0
    normalized_w[mask] = torch.sign(w)[mask] * normalized_w[mask]
    weights = normalized_w.float().cuda()

    found = False
    i = 0
    while not found and i < len(test_dset):
        _, garment_id, pcd, _, _, _ = test_dset[i]
        found = garment_id == required_garment_id
        i += 1

    if not found:
        raise ValueError(f"Garment {required_garment_id} not found in test set")

    pcd = pcd.unsqueeze(0).cuda()
    pcd = random_point_sampling(pcd, num_points_pcd)

    with torch.no_grad():
        latent_code = encoder(pcd)

    latent_codes = [latent_code]
    for i in range(1, num_steps + 1):
        step = weights * (i * step_size)
        new_code = latent_code + step
        latent_codes.append(new_code)

    for i in progress_bar(range(len(latent_codes)), "Meshing"):
        lat = latent_codes[i]

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
        mesh_path = get_out_dir() / f"editing_{garment_id}/{i}.obj"
        mesh_path.parent.mkdir(exist_ok=True, parents=True)
        o3d.io.write_triangle_mesh(str(mesh_path), pred_mesh_o3d)


if __name__ == "__main__":
    main()
