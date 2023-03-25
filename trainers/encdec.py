from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from hesiod import get_out_dir, hcfg
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.cloth3d import Cloth3d
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import compute_gradients, progress_bar, random_point_sampling


class EncoderDecoderTrainer:
    def __init__(self) -> None:
        train_ids_file = Path(hcfg("dset.train_ids_file", str))
        dset_category = hcfg("dset.category", str)
        dset_root = Path(hcfg("dset.root", str))

        train_dset = Cloth3d(train_ids_file, dset_root, dset_category)
        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(
            train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
        )

        self.num_points_pcd = hcfg("num_points_pcd", int)
        latent_size = hcfg("latent_size", int)
        self.max_dist = hcfg("udf_max_dist", float)
        self.num_points_forward = hcfg("num_points_forward", int)

        encoder = Dgcnn(latent_size)
        self.encoder = encoder.cuda()

        self.coords_encoder = CoordsEncoder()

        decoder_cfg = hcfg("decoder", Dict[str, Any])
        decoder = CbnDecoder(
            self.coords_encoder.out_dim,
            latent_size,
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers"],
        )
        self.decoder = decoder.cuda()

        params = list(encoder.parameters())
        params.extend(decoder.parameters())
        lr = hcfg("lr", float)
        self.optimizer = Adam(params, lr)

        self.epoch = 0
        self.global_step = 0

        self.ckpts_path = get_out_dir() / "ckpts"

        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)

        self.logger = SummaryWriter(get_out_dir() / "logs")

    def train(self) -> None:
        num_epochs = hcfg("num_epochs", int)
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            desc = f"Epoch {epoch}/{num_epochs}"
            for batch in progress_bar(self.train_loader, desc=desc):
                _, _, pcds, coords, gt_udf, gt_grad = batch
                pcds = pcds.cuda()
                coords = coords.cuda()
                gt_udf = gt_udf.cuda()
                gt_grad = gt_grad.cuda()

                pcds = random_point_sampling(pcds, self.num_points_pcd)

                gt_udf = gt_udf / self.max_dist
                gt_udf = 1 - gt_udf
                c_u_g = torch.cat([coords, gt_udf.unsqueeze(-1), gt_grad], dim=-1)

                selected_c_u_g = random_point_sampling(c_u_g, self.num_points_forward)
                selected_coords = selected_c_u_g[:, :, :3]
                selected_coords.requires_grad = True
                selected_gt_udf = selected_c_u_g[:, :, 3]
                selected_gt_grad = selected_c_u_g[:, :, 4:]

                latent_codes = self.encoder(pcds)
                coords_encoded = self.coords_encoder.encode(selected_coords)
                pred = self.decoder(coords_encoded, latent_codes)

                udf_loss = F.binary_cross_entropy_with_logits(pred, selected_gt_udf)

                udf_pred = torch.sigmoid(pred)
                udf_pred = 1 - udf_pred
                udf_pred *= self.max_dist
                gradients = compute_gradients(selected_coords, udf_pred)

                grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")
                mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
                grad_loss = grad_loss[mask].mean()

                self.optimizer.zero_grad()

                loss = udf_loss + 0.1 * grad_loss

                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0:
                    self.logger.add_scalar(
                        "train/udf_loss",
                        udf_loss.item(),
                        self.global_step,
                    )
                    self.logger.add_scalar(
                        "train/grad_loss",
                        grad_loss.item(),
                        self.global_step,
                    )

                self.global_step += 1

            if epoch % 50 == 49:
                self.save_ckpt(all=True)

            self.save_ckpt()

    def save_ckpt(self, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if all:
            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)
        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "last" in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"last_{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "last" in p.name]
            error_msg = "Expected only one last ckpt, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
