import numpy as np
import torch
import trimesh
import os
import mesh2sdf
import time
from scipy.spatial.transform import Rotation as R

QUASI_DYN_ROOT = "/home/xueyi/diffsim/NeuS"
if not os.path.exists(QUASI_DYN_ROOT):
    QUASI_DYN_ROOT = "/root/diffsim/quasi-dyn"

ARCTIC_CANON_OBJ_SV_FOLDER = os.path.join(QUASI_DYN_ROOT, "raw_data/arctic_processed_canon_obj")



def export_canon_obj_file(kinematic_mano_gt_sv_fn, obj_name):
    
    subject_idx = kinematic_mano_gt_sv_fn.split("/")[-2] # 
    print(f"subject_idx: {subject_idx}, obj name: {obj_name}")
    sv_dict = np.load(kinematic_mano_gt_sv_fn, allow_pickle=True).item()
    
    
    object_global_orient = sv_dict["obj_rot"]  
    object_transl = sv_dict["obj_trans"] * 0.001
    obj_pcs = sv_dict["verts.object"]
    
    # obj_pcs = sv_dict['object_pc']
    obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    
    # self.obj_verts = obj_verts
    init_obj_verts = obj_pcs[0]
    init_obj_rot_vec = object_global_orient[0]
    init_obj_transl = object_transl[0]
    
    init_obj_transl = torch.from_numpy(init_obj_transl).float().cuda()
    init_rot_struct = R.from_rotvec(init_obj_rot_vec)
    
    init_glb_rot_mtx = init_rot_struct.as_matrix()
    init_glb_rot_mtx = torch.from_numpy(init_glb_rot_mtx).float().cuda()
    # ## reverse the global rotation matrix ##
    init_glb_rot_mtx_reversed = init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()


    
    ''' canonical object verts '''
    canon_obj_verts = torch.matmul(
        init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
    ).contiguous().transpose(1, 0).contiguous()

    # ## get canon obj verts ##
    
    # /home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj
    
    canon_obj_sv_folder = ARCTIC_CANON_OBJ_SV_FOLDER # "/root/diffsim/control-vae-2/assets/arctic"
    canon_obj_mesh = trimesh.Trimesh(vertices=canon_obj_verts.detach().cpu().numpy(), faces=sv_dict['f'][0])
    
    canon_obj_mesh_sv_fn = f"{subject_idx}_{obj_name}.obj"
    canon_obj_mesh_sv_fn = os.path.join(canon_obj_sv_folder, canon_obj_mesh_sv_fn)
    canon_obj_mesh.export(canon_obj_mesh_sv_fn)

    print(f"Canonical obj mesh saved to {canon_obj_mesh_sv_fn}")
    return canon_obj_mesh_sv_fn


def compute_sdf(obj_file_name):
    filename = obj_file_name

    # init_mesh_scale = 1.0
    init_mesh_scale = 0.8

    mesh_scale = 0.8
    size = 128
    level = 2 / size

    mesh = trimesh.load(filename, force='mesh')

    # normalize mesh
    vertices = mesh.vertices
    vertices = vertices * init_mesh_scale
    bbmin = vertices.min(0) # 
    bbmax = vertices.max(0) # 
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max # # bbmax - bbmin # 
    vertices = (vertices - center) * scale # (vertices - center) * scale #

    scaled_bbmin = vertices.min(0)
    scaled_bbmax = vertices.max(0)
    print(f"scaled_bbmin: {scaled_bbmin}, scaled_bbmax: {scaled_bbmax}")


    t0 = time.time()
    sdf, mesh = mesh2sdf.compute( ## sdf and mesh ##
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    t1 = time.time()

    print(f"sdf: {sdf.shape}, mesh: {mesh.vertices.shape}")

    mesh.vertices = mesh.vertices / scale + center
    mesh.export(filename[:-4] + '.fixed.obj') ## .fixed.obj ##
    np.save(filename[:-4] + '.npy', sdf) ## .npy ##
    print('It takes %.4f seconds to process %s' % (t1-t0, filename))


if __name__=='__main__':
    
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s01/box_grab_01.npy'
    
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s01/ketchup_grab_01.npy'
    
    # kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s01/mixer_grab_01.npy'
    
    # kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s01/laptop_grab_01.npy'
    
    # kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s01/box_grab_01.npy'
    
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s02/waffleiron_grab_01.npy'
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s02/ketchup_grab_01.npy'
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s02/phone_grab_01.npy'
    kinematic_mano_gt_sv_fn = '/data/xueyi/sim/arctic_processed_data/processed_seqs/s02/box_grab_01.npy'
    
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s02"
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s04"
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s05"
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s06"
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s07"
    kinematic_mano_gt_sv_folder = "/data/xueyi/sim/arctic_processed_data/processed_seqs/s10"
    
    kinematic_mano_gt_sv_fn_all = os.listdir(kinematic_mano_gt_sv_folder)
    kinematic_mano_gt_sv_fn_all = [fn for fn in kinematic_mano_gt_sv_fn_all if fn.endswith(".npy") and "grab" in fn]
    kinematic_mano_gt_sv_fn_all = [os.path.join(kinematic_mano_gt_sv_folder, fn) for fn in kinematic_mano_gt_sv_fn_all]
    for kinematic_mano_gt_sv_fn in kinematic_mano_gt_sv_fn_all:
        obj_name = kinematic_mano_gt_sv_fn.split("/")[-1].split(".")[0].split("_")[0]
        print(f"obj_name: {obj_name}")
        
        ##### process and export canonical object file #####
        canon_obj_mesh_sv_fn = export_canon_obj_file(kinematic_mano_gt_sv_fn, obj_name)
        
        
        ##### compute sdf of the canonical object mesh #####
        compute_sdf(canon_obj_mesh_sv_fn)
        
    
    # obj_name = kinematic_mano_gt_sv_fn.split("/")[-1].split(".")[0].split("_")[0]
    # print(f"obj_name: {obj_name}")
    
    # ##### process and export canonical object file #####
    # canon_obj_mesh_sv_fn = export_canon_obj_file(kinematic_mano_gt_sv_fn, obj_name)
    
    
    # ##### compute sdf of the canonical object mesh #####
    # compute_sdf(canon_obj_mesh_sv_fn)
    
    
    