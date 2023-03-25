import os
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from models import drapenet, coordsenc, cbndec
from meshudf.meshudf import get_mesh_from_udf

def load_udf(checkpoints_dir, code_file_name, model_file_name, device):
    hidden_dim = 512
    num_hidden_layers = 5
    latent_size = 32

    coords_encoder = coordsenc.CoordsEncoder()

    decoder = cbndec.CbnDecoder(
        coords_encoder.out_dim,
        latent_size,
        hidden_dim,
        num_hidden_layers,
    )

    latent_code = torch.load(os.path.join(checkpoints_dir, code_file_name)).to(device)
    decoder.load_state_dict(torch.load(os.path.join(checkpoints_dir, model_file_name))["decoder"])
    decoder = decoder.to(device)
    decoder.eval()

    return coords_encoder, latent_code, decoder

def reconstruct(coords_encoder, decoder, lat, udf_max_dist=0.1, resolution=256, differentiable=False):
    def udf_func(c):
        c = coords_encoder.encode(c.unsqueeze(0))
        p = decoder(c, lat).squeeze(0)
        p = torch.sigmoid(p)
        p = (1 - p) * udf_max_dist
        return p

    v, t = get_mesh_from_udf(
        udf_func,
        coords_range=(-1, 1),
        max_dist=udf_max_dist,
        N=resolution,
        max_batch=2**16,
        differentiable=differentiable,
        use_fast_grid_filler=True
    )
    
    mesh = trimesh.Trimesh(v.squeeze().cpu().numpy(), t.squeeze().cpu().numpy(), process=False, valid=False)
    return mesh, v, t

def load_lbs(checkpoints_dir, device):
    embedder, embed_dim = drapenet.get_embedder(4)

    model_lbs = drapenet.skip_connection(d_in=3, d_out=24, width=256, depth=8, skip_layer=[4]).to(device).eval()
    model_lbs.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lbs.pth')))

    model_lbs_shape = drapenet.skip_connection(d_in=10+3, d_out=3, width=256, depth=8, skip_layer=[4]).to(device).eval()
    model_lbs_shape.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lbs_shape.pth')))

    model_lbs_deform_top = drapenet.lbs_pbs(d_in_theta=72 + 10, d_in_x=embed_dim+32, d_out_p=128, hidden_theta=512, hidden_matrix=512, skip=False, soft_max=False, init=False).to(device).eval()
    model_lbs_deform_top.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lbs_deform_top.pth')))

    model_lbs_deform_bottom = drapenet.lbs_pbs(d_in_theta=72 + 10, d_in_x=embed_dim+32, d_out_p=128, hidden_theta=512, hidden_matrix=512, skip=False, soft_max=False, init=False).to(device).eval()
    model_lbs_deform_bottom.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lbs_deform_bottom.pth')))

    model_lbs_deform_layer = drapenet.lbs_pbs(d_in_theta=72 + 10, d_in_x=embed_dim+32*2, d_out_p=128, hidden_theta=512, hidden_matrix=512, skip=False, soft_max=False, init=False).to(device).eval()
    model_lbs_deform_layer.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'lbs_deform_layer.pth')))

    return embedder, model_lbs, model_lbs_shape, model_lbs_deform_top, model_lbs_deform_bottom, model_lbs_deform_layer

def skinning(x, w, tfs, tfs_c_inv):
    """Linear blend skinning
    Args:
        x (tensor): deformed points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        tfs_c_inv (tensor): bone transformation matrices. shape: [J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    tfs = torch.einsum('bnij,njk->bnik', tfs, tfs_c_inv)

    x_h = F.pad(x, (0, 1), value=1.0)
    x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :3]

def deforming(vertices_garment_T, faces, pose, beta, latent_code, embedder, model_lbs, model_lbs_shape, model_lbs_deform, tfs, tfs_c_inv):
    # vertices_garment_T - (#P, 3) 
    # tfs - (1, 24, 4, 4) 
    # tfs_c_inv - (24, 4, 4) 
    # pose - (1, 72) 
    # beta - (1, 10) 
    # latent_code - (1, 32) 

    num_v = vertices_garment_T.shape[0]
    points = vertices_garment_T.unsqueeze(0)
    points_embed = embedder(points*5)
        
    latent_code = latent_code.unsqueeze(1).repeat(1, num_v, 1)
    points_embed = torch.cat((points_embed, latent_code), dim=-1)
        
    pose_input = pose.unsqueeze(1).repeat(1, num_v, 1)
    beta_input = beta.unsqueeze(1).repeat(1, num_v, 1)

    input_lbs_deform = torch.cat((pose_input, beta_input), dim=-1)
    
    x_deform = model_lbs_deform(input_lbs_deform, points_embed)/100
    garment_deform = points + x_deform

    input_lbs_shape = torch.cat((points, beta_input), dim=-1)
    delta_shape_pred = model_lbs_shape(input_lbs_shape)
    garment_deform += delta_shape_pred

    lbs_weight = model_lbs(points)
    lbs_weight = lbs_weight.softmax(dim=-1)
    garment_skinning = skinning(garment_deform, lbs_weight, tfs, tfs_c_inv)

    verts_deformed = garment_skinning.squeeze() # (#P, 3) 
    verts_deformed_np = verts_deformed.detach().cpu().numpy()

    cloth_mesh = trimesh.Trimesh(verts_deformed_np, faces)
    return verts_deformed, cloth_mesh

def deforming_layer(vertices_garment_T, faces, pose, beta, latent_code, latent_code_top, embedder, model_lbs, model_lbs_shape, model_lbs_deform, model_lbs_layer, tfs, tfs_c_inv):
    # vertices_garment_T - (#P, 3) 
    # tfs - (1, 24, 4, 4) 
    # tfs_c_inv - (24, 4, 4) 
    # pose - (1, 72) 
    # beta - (1, 10) 
    # latent_code - (1, 32) 
    # latent_code_top - (1, 32) 

    num_v = vertices_garment_T.shape[0]
    points = vertices_garment_T.unsqueeze(0)
    points_embed = embedder(points*5)
        
    latent_code = latent_code.unsqueeze(1).repeat(1, num_v, 1)
    points_embed = torch.cat((points_embed, latent_code), dim=-1)
        
    pose_input = pose.unsqueeze(1).repeat(1, num_v, 1)
    beta_input = beta.unsqueeze(1).repeat(1, num_v, 1)

    input_lbs_deform = torch.cat((pose_input, beta_input), dim=-1)
    
    x_deform = model_lbs_deform(input_lbs_deform, points_embed)/100
    garment_deform = points + x_deform
    
    input_lbs_shape = torch.cat((points, beta_input), dim=-1)
    delta_shape_pred = model_lbs_shape(input_lbs_shape)
    garment_deform += delta_shape_pred

    layer_top_latent_code = latent_code_top.unsqueeze(1).repeat(1, num_v, 1)
    layer_embed = torch.cat((points_embed, layer_top_latent_code), dim=-1)
    
    x_deform_layer = model_lbs_layer(input_lbs_deform, layer_embed)#/100
    x_deform_layer /= 100
    garment_deform_with_layer = garment_deform + x_deform_layer

    lbs_weight = model_lbs(points)
    lbs_weight = lbs_weight.softmax(dim=-1)
    garment_skinning = skinning(garment_deform, lbs_weight, tfs, tfs_c_inv)
    garment_skinning_with_layer = skinning(garment_deform_with_layer, lbs_weight, tfs, tfs_c_inv)

    verts_deformed = garment_skinning.squeeze() # (#P, 3) 
    verts_deformed_layer = garment_skinning_with_layer.squeeze() # (#P, 3) 
    verts_deformed_np = verts_deformed.detach().cpu().numpy()
    verts_deformed_layer_np = verts_deformed_layer.detach().cpu().numpy()

    cloth_mesh = trimesh.Trimesh(verts_deformed_np, faces)
    cloth_mesh_layer = trimesh.Trimesh(verts_deformed_layer_np, faces)
    return verts_deformed, verts_deformed_layer, cloth_mesh, cloth_mesh_layer

def draping(vertices_Ts, faces_garments, latent_codes, pose, beta, models, smpl_server, tfs_c_inv):
    vertices_top_T, vertices_bottom_T = vertices_Ts
    faces_top, faces_bottom = faces_garments
    latent_code_top, latent_code_bottom = latent_codes
    embedder, _lbs, _lbs_shape, _lbs_deform_top, _lbs_deform_bottom, _lbs_deform_layer = models
    with torch.no_grad():
        output_smpl = smpl_server(betas=beta, body_pose=pose[:, 3:], global_orient=pose[:, :3], return_verts=True)
        tfs = output_smpl.T
        smpl_verts = output_smpl.vertices

        _, top_mesh = deforming(vertices_top_T, faces_top, pose, beta, latent_code_top, embedder, _lbs, _lbs_shape, _lbs_deform_top, tfs, tfs_c_inv)

        _, _, bottom_mesh, bottom_mesh_layer = deforming_layer(vertices_bottom_T, faces_bottom, pose, beta, latent_code_bottom, latent_code_top, embedder, _lbs, _lbs_shape, _lbs_deform_bottom, _lbs_deform_layer, tfs, tfs_c_inv)

    body_mesh = trimesh.Trimesh(smpl_verts.squeeze().cpu().numpy(), smpl_server.faces)

    colors_f_body = np.ones((len(body_mesh.faces), 4))*np.array([255, 255, 255, 200])[np.newaxis,:]
    colors_f_top = np.ones((len(top_mesh.faces), 4))*np.array([160, 160, 255, 200])[np.newaxis,:]
    colors_f_bottom = np.ones((len(bottom_mesh.faces), 4))*np.array([100, 100, 100, 200])[np.newaxis,:]
    body_mesh.visual.face_colors = colors_f_body
    top_mesh.visual.face_colors = colors_f_top
    bottom_mesh.visual.face_colors = colors_f_bottom
    bottom_mesh_layer.visual.face_colors = colors_f_bottom
        
    return top_mesh, bottom_mesh, bottom_mesh_layer, body_mesh