

import numpy as np
import trimesh
import os 
# try:
#     import mesh2sdf
# except:
#     pass
import mesh2sdf
import time
from scipy.spatial.transform import Rotation as R

def extract_obj_meshes(sv_dict_fn):
    # sv_fn = "/data2/datasets/sim/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy"
    # if not os.path.exists(sv_fn):
    #     sv_fn = "/data/xueyi/arctic/processed_sv_dicts/box_grab_01_extracted_dict.npy"
    active_passive_sv_dict = np.load(sv_dict_fn, allow_pickle=True).item() # 

    obj_verts = active_passive_sv_dict['obj_verts'] # object orientation #
    obj_faces = active_passive_sv_dict['obj_faces']
    
    obj_mesh = trimesh.Trimesh(obj_verts[0], obj_faces)
    obj_mesh_sv_fn = os.path.join(
        "/home/xueyi/diffsim/NeuS/utils", "init_box.obj"
    )
    obj_mesh.export(obj_mesh_sv_fn)
    
    obj_verts_reversed = obj_verts[:, :, [1, 0, 2]]
    obj_verts_reversed[:, :, 1] = -obj_verts_reversed[:, :, 1]
    obj_mesh_reversed = trimesh.Trimesh(obj_verts_reversed[0], obj_faces)
    obj_mesh_sv_fn_reversed = os.path.join(
        "/home/xueyi/diffsim/NeuS/utils", "init_box_reversed.obj"
    )
    obj_mesh_reversed.export(obj_mesh_sv_fn_reversed)


def extract_obj_meshes_boundingbox(sv_dict_fn):
    # sv_fn = "/data2/datasets/sim/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy"
    # if not os.path.exists(sv_fn):
    #     sv_fn = "/data/xueyi/arctic/processed_sv_dicts/box_grab_01_extracted_dict.npy"
    active_passive_sv_dict = np.load(sv_dict_fn, allow_pickle=True).item() # 

    obj_verts = active_passive_sv_dict['obj_verts'] # object orientation #
    obj_faces = active_passive_sv_dict['obj_faces']
    
    init_obj_verts = obj_verts[0]
    minn_box = np.min(init_obj_verts, axis=0, keepdims=True)
    maxx_box = np.max(init_obj_verts, axis=0, keepdims=True)
    # get the minn box and maxx box #
    
    
    box_triangle_mesh_faces = np.array([
        [1, 2, 3],  # Left face (triangle 1)
        [2, 3, 4],  # Left face (triangle 2)
        [5, 6, 7],  # Right face (triangle 1)
        [6, 7, 8],  # Right face (triangle 2)
        [1, 3, 5],  # Bottom face (triangle 1)
        [3, 5, 7],  # Bottom face (triangle 2)
        [2, 4, 6],  # Top face (triangle 1)
        [4, 6, 8],  # Top face (triangle 2)
        [1, 2, 5],  # Front face (triangle 1)
        [2, 5, 6],  # Front face (triangle 2)
        [3, 4, 7],  # Back face (triangle 1)
        [4, 7, 8]   # Back face (triangle 2)
    ], dtype=np.int32) - 1
    
    box_vertices = np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ], dtype=np.float32)
    
    box_vertices = (box_vertices -  (-1)) / 2
    
    
    box_vertices = box_vertices * (maxx_box - minn_box) + minn_box
    
    box_mesh=  trimesh.Trimesh(box_vertices, box_triangle_mesh_faces)
    # 
    
    # obj_mesh = trimesh.Trimesh(obj_verts[0], obj_faces)
    obj_mesh_sv_fn = os.path.join(
        "/home/xueyi/diffsim/NeuS/utils", "init_bounding_box.obj"
    )
    box_mesh.export(obj_mesh_sv_fn)
    
    # obj_verts_reversed = obj_verts[:, :, [1, 0, 2]]
    # obj_verts_reversed[:, :, 1] = -obj_verts_reversed[:, :, 1]
    # obj_mesh_reversed = trimesh.Trimesh(obj_verts_reversed[0], obj_faces)
    # obj_mesh_sv_fn_reversed = os.path.join(
    #     "/home/xueyi/diffsim/NeuS/utils", "init_box_reversed.obj"
    # )
    # obj_mesh_reversed.export(obj_mesh_sv_fn_reversed)
    
def extract_obj_meshes_taco(pkl_fn):
    import pickle as pkl
    
    sv_dict = pkl.load(open(pkl_fn, "rb"))
    
    print(f"sv_dict: {sv_dict.keys()}")
    
    # maxx_ws = min(maxx_ws, len(sv_dict['obj_verts']) - start_idx)
    
    obj_pcs = sv_dict['obj_verts'] # [start_idx: start_idx + maxx_ws]
    # obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    # self.obj_pcs = obj_pcs
    # # obj_vertex_normals = sv_dict['obj_vertex_normals']
    # # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
    # self.obj_normals = torch.zeros_like(obj_pcs[0]) ### get the obj naormal vectors ##
    
    object_pose = sv_dict['obj_pose'] # [start_idx: start_idx + maxx_ws]
    # object_pose = torch.from_numpy(object_pose).float().cuda() ### nn_frames x 4 x 4 ###
    object_global_orient_mtx = object_pose[:, :3, :3 ] ## nn_frames x 3 x 3 ##
    object_transl = object_pose[:, :3, 3] ## nn_frmaes x 3 ##
    
    obj_faces = sv_dict['obj_faces']
    # obj_faces = torch.from_numpy(obj_faces).long().cuda()
    # self.obj_faces = obj_faces # [0] ### obj faces ##
    
    # obj_verts = sv_dict['obj_verts']
    # minn_verts = np.min(obj_verts, axis=0)
    # maxx_verts = np.max(obj_verts, axis=0)
    # extent = maxx_verts - minn_verts
    # center_ori = (maxx_verts + minn_verts) / 2
    # scale_ori = np.sqrt(np.sum(extent ** 2))
    # obj_verts = torch.from_numpy(obj_verts).float().cuda()
    
    init_obj_verts = obj_pcs[0]
    init_obj_ornt_mtx = object_global_orient_mtx[0]
    init_obj_transl = object_transl[0]
    
    canon_obj_verts = np.matmul(
        init_obj_ornt_mtx.T, (init_obj_verts - init_obj_transl[None]).T
    ).T
    # self.obj_verts = canon_obj_verts.clone()
    # obj_verts = canon_obj_verts.clone()
    canon_obj_mesh = trimesh.Trimesh(vertices=canon_obj_verts, faces=obj_faces)
    canon_obj_mesh_export_dir = "/".join(pkl_fn.split("/")[:-1])
    pkl_name = pkl_fn.split("/")[-1].split(".")[0]
    canon_obj_mesh_sv_fn = f"{pkl_name}.obj"
    canon_obj_mesh.export(os.path.join(canon_obj_mesh_export_dir, canon_obj_mesh_sv_fn))
    print(f"canon_obj_mesh_sv_fn: {canon_obj_mesh_sv_fn}")
        
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
    
    # compute_sdf('/data/xueyi/taco/processed_data/20230917/right_20230917_032.obj')
    # exit(0)
    
    pkl_fn = "/data3/datasets/xueyi/taco/processed_data/20230917/right_20230917_004.pkl"
    pkl_fn = "/data3/datasets/xueyi/taco/processed_data/20230917/right_20230917_030.pkl"
    pkl_fn = "/data3/datasets/xueyi/taco/processed_data/20230917/right_20230917_037.pkl"
    pkl_fn = "/data/xueyi/taco/processed_data/20230917/right_20230917_037.pkl"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20230917"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231010"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20230919"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231104"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231105"
    pkl_root_folder = "/data/xueyi/taco/processed_data/20230917" ## pkl folder 
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231102"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20230923"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20230926"
    pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20230927"
    pkl_root_folder = "/data/xueyi/taco/processed_data/20230930"
    pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20231031"
    # /data2/datasets/xueyi/taco/processed_data/20231027
    pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20231027"
    pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20231026"
    pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20231024"
    pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20231020"
    # pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20230929"
    # pkl_root_folder = "/data2/datasets/xueyi/taco/processed_data/20230930"
    tot_pkl_fns = os.listdir(pkl_root_folder)
    tot_pkl_fns  = [fn for fn in tot_pkl_fns if fn.endswith(".pkl")]
    # tot_pkl_fns  = ['right_20230930_001.pkl']
    # tot_pkl_fns = ['/data/xueyi/taco/processed_data/20230917/right_20230917_032.obj']
    for i_fn, cur_pkl_fn in enumerate(tot_pkl_fns):
        cur_full_pkl_fn = os.path.join(pkl_root_folder, cur_pkl_fn)
        extract_obj_meshes_taco(cur_full_pkl_fn)
        # compute_sdf(cur_full_fn)
    # exit(0)
    
    # pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231104"
    # pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231105"
    # pkl_root_folder = "/data/xueyi/taco/processed_data/20230917"
    # pkl_root_folder = "/data3/datasets/xueyi/taco/processed_data/20231102"
    tot_fns = os.listdir(pkl_root_folder)
    tot_fns = [fn for fn in tot_fns if fn.endswith(".obj")]
    for cur_fn in tot_fns:
        cur_full_fn = os.path.join(pkl_root_folder, cur_fn)
        compute_sdf(cur_full_fn)
    
    # obj_mesh_fn = "/data3/datasets/xueyi/taco/processed_data/20231104/right_20231104_017.obj"
    # compute_sdf(obj_mesh_fn)
    exit(0)
    
    extract_obj_meshes_taco(pkl_fn)
    exit(0)
    
    sv_dict_fn = "/data2/datasets/sim/arctic_processed_data/processed_sv_dicts/s01/box_grab_01_extracted_dict.npy"
    # extract_obj_meshes(sv_dict_fn=sv_dict_fn)
    
    extract_obj_meshes_boundingbox(sv_dict_fn)
        

