import sys

sys.path.append("..")

from pathlib import Path

import numpy as np
from utils import (
    compute_udf_from_mesh,
    get_o3d_mesh_from_tensors,
    get_tensor_pcd_from_o3d,
    progress_bar,
    read_mesh,
)

if len(sys.argv) != 3:
    print("Usage: python3 preprocess_udf.py </path/to/meshes> </out/path>")
    exit(1)

meshes_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

meshes_paths = list(meshes_dir.glob("*.obj"))

for mesh_path in progress_bar(meshes_paths):
    v, t = read_mesh(mesh_path)
    mesh_o3d = get_o3d_mesh_from_tensors(v, t)

    pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=100_000)
    pcd = get_tensor_pcd_from_o3d(pcd_o3d)[:, :3]

    coords, labels, gradients = compute_udf_from_mesh(
        mesh_o3d,
        num_queries_on_surface=250_000,
        num_queries_per_std=[250_000, 200_000, 25_000, 25_000],
    )

    out_file = out_dir / f"{mesh_path.stem}.npz"
    np.savez(
        out_file,
        vertices=v.numpy(),
        triangles=t.numpy(),
        pcd=pcd.numpy(),
        coords=coords.numpy(),
        labels=labels.numpy(),
        gradients=gradients.numpy(),
    )
