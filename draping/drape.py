import os, sys
import numpy as np
import trimesh
import torch

sys.path.append("..")
from smpl_pytorch.body_models import SMPL
from utils_drape import draping, load_udf, load_lbs, reconstruct


def main(pose, beta, checkpoints_dir, extra_dir, smpl_model_dir, output_folder, device, top_idx=0, bottom_idx=0, resolution=256):
    ''' Load pretrained models '''
    models = load_lbs(checkpoints_dir, device)
    _, latent_codes_top, decoder_top = load_udf(checkpoints_dir, 'top_codes.pt', 'top_udf.pt', device)
    coords_encoder, latent_codes_bottom, decoder_bottom = load_udf(checkpoints_dir, 'bottom_codes.pt', 'bottom_udf.pt', device)

    ''' Initialize SMPL model '''
    data_body = np.load(os.path.join(extra_dir, 'body_info_f.npz'))
    tfs_c_inv = torch.FloatTensor(data_body['tfs_c_inv']).to(device)
    smpl_server = SMPL(model_path=smpl_model_dir, gender='f').to(device)

    ''' Reconstruct shirt(top_idx)/pants(bottom_idx) in T-psoe '''
    mesh_top, vertices_top_T, faces_top = reconstruct(coords_encoder, decoder_top, latent_codes_top[[top_idx]], udf_max_dist=0.1, resolution=resolution, differentiable=False)
    mesh_top.export(output_folder + '/top-T.obj')
    mesh_bottom, vertices_bottom_T, faces_bottom = reconstruct(coords_encoder, decoder_bottom, latent_codes_bottom[[bottom_idx]], udf_max_dist=0.1, resolution=resolution, differentiable=False)
    mesh_bottom.export(output_folder + '/bottom-T.obj')

    ''' Skinning garments '''
    vertices_Ts = [vertices_top_T, vertices_bottom_T]
    faces_garments = [faces_top.cpu().numpy(), faces_bottom.cpu().numpy()]
    latent_codes = [latent_codes_top[[top_idx]], latent_codes_bottom[[bottom_idx]]]

    top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh = draping(vertices_Ts, faces_garments, latent_codes, pose, beta, models, smpl_server, tfs_c_inv)
    body_mesh.export(output_folder + '/body.obj')
    top_mesh.export(output_folder + '/shirt.obj')
    bottom_mesh.export(output_folder + '/pants.obj')
    bottom_mesh_layer.export(output_folder + '/pants-layer.obj')


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    ''' dir to load models and extra data '''
    checkpoints_dir = '../checkpoints'
    extra_dir = '../extra-data'
    smpl_model_dir = '../smpl_pytorch'

    ''' dir to dump mesh '''
    output_folder = '../output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ''' Example SMPL parameters '''
    poses = torch.load(os.path.join(extra_dir, 'pose-sample.pt'))
    pose = poses[[0]].to(device)
    beta = torch.zeros(1, 10).to(device)

    ''' top_idx/bottom_idx - the index corresponding to the shirt/pants '''
    main(pose, beta, checkpoints_dir, extra_dir, smpl_model_dir, output_folder, device, top_idx=208, bottom_idx=15, resolution=256)