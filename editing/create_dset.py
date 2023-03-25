import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import List

from hesiod import hcfg, hmain

from data.cloth3d import Cloth3d
from utils import get_o3d_mesh_from_tensors, read_mesh

import open3d as o3d  # isort: skip


@hmain(
    base_cfg_dir="../cfg/bases",
    template_cfg_file="../cfg/editing_dset.yaml",
    create_out_dir=False,
)
def main() -> None:
    ids_file = Path(hcfg("dset.train_ids_file", str))
    root = Path(hcfg("dset.root", str))
    category = hcfg("dset.category", str)
    dset = Cloth3d(ids_file, root, category)

    v, t = read_mesh(hcfg("dset.tposed_body_mesh", str))
    body_o3d = get_o3d_mesh_from_tensors(v, t)

    out_file = Path(hcfg("out_file", str))
    nums = hcfg("num_samples", List[int])
    classes = hcfg("classes", List[str])
    keys = hcfg("keys", List[int])

    print(
        f"Press {chr(keys[0]).upper()} to mark as {classes[0].upper()} "
        f"and {chr(keys[1]).upper()} to mark as {classes[1].upper()}. "
        "Press Q to skip."
    )

    garment_id = 0
    class_0_saved_ids = []
    class_1_saved_ids = []
    seen_ids = []

    def save_0(vis) -> None:
        if len(class_0_saved_ids) < nums[0]:
            with open(out_file, "at") as f:
                f.write(f"{garment_id},{classes[0]}\n")
            class_0_saved_ids.append(garment_id)
        vis.close()

    def save_1(vis) -> None:
        if len(class_1_saved_ids) < nums[1]:
            with open(out_file, "at") as f:
                f.write(f"{garment_id},{classes[1]}\n")
            class_1_saved_ids.append(garment_id)
        vis.close()

    while len(class_0_saved_ids) != nums[0] or len(class_1_saved_ids) != nums[1]:
        print(
            f"{classes[0].upper()}: {len(class_0_saved_ids)}/{nums[0]} - "
            f"{classes[1].upper()}: {len(class_1_saved_ids)}/{nums[1]}"
        )
        idx = randint(0, len(dset) - 1)

        _, garment_id, _, _, _, _ = dset[idx]
        if garment_id in seen_ids:
            continue
        seen_ids.append(garment_id)

        v, t = dset.get_mesh(idx)

        mesh_o3d = get_o3d_mesh_from_tensors(v, t)
        mesh_o3d.paint_uniform_color((0, 0, 1))

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("garments", 800, 800)
        vis.add_geometry(body_o3d)
        vis.add_geometry(mesh_o3d)

        vis.register_key_callback(keys[0], save_0)
        vis.register_key_callback(keys[1], save_1)

        vis.run()


if __name__ == "__main__":
    main()
