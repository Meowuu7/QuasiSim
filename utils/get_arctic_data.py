

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import torch
from manopth.manolayer import ManoLayer
import trimesh
import dyn_model_act_v2 as dyn_model_act_mano

def batched_get_orientation_matrices(rot_vec):
    rot_matrices = []
    for i_w in range(rot_vec.shape[0]):
        cur_rot_vec = rot_vec[i_w]
        cur_rot_mtx = R.from_rotvec(cur_rot_vec).as_matrix()
        rot_matrices.append(cur_rot_mtx)
    rot_matrices = np.stack(rot_matrices, axis=0)
    return rot_matrices


def extract_hand_obj_data_tst(seq_path, split='test'):
    # for i_f, f in enumerate(files_clean):
    # if split == 'train':
    #     print(f"loading {i_f} / {len(files_clean)}")
    # if split != 'train' and i_f >= 100:
    #     break
    # if args is not None and args.debug and i_f >= 10:
    #     break
    mano_path = "/data1/xueyi/mano_models/mano/models"
    
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_root #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    )
    
    # save_dict_dir = "/data1/xueyi/GRAB_extracted_test" ## 
    # save_dict_dir = os.path.join(save_dict_dir, split) ## extrac
    # if not os.path.exists(save_dict_dir):
    #     os.makedirs(save_dict_dir)
    
    subj_data_folder = '/data1/xueyi/GRAB_processed_wsubj'
    clip_clean = np.load(seq_path)

    # print(f"clip_clean: {clip_clean.keys()}")
    # for k in clip_clean:
    
    object_index = clip_clean['f7'][0].item()
    print(f"object_index: {object_index}")
    
    
    ## get the grab obj ###
    grab_path = "/data1/xueyi/GRAB_extracted"
    obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes') ## one index shift here ##
    id2objmesh = []
    ## obj meshes ##
    obj_meshes = sorted(os.listdir(obj_mesh_path))
    for i, fn in enumerate(obj_meshes):
        # id2objmesh.append(os.path.join(obj_mesh_path, fn))
        id2objmesh.append(fn)
    print(id2objmesh)


def extract_hand_obj_data(seq_path, split='test', window_size=None):
    # for i_f, f in enumerate(files_clean):
    # if split == 'train':
    #     print(f"loading {i_f} / {len(files_clean)}")
    # if split != 'train' and i_f >= 100:
    #     break
    # if args is not None and args.debug and i_f >= 10:
    #     break
    mano_path = "/data1/xueyi/mano_models/mano/models"
    
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_root #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    )
    
    save_dict_dir = "/data1/xueyi/GRAB_extracted_test" ## 
    save_dict_dir = os.path.join(save_dict_dir, split)
    if not os.path.exists(save_dict_dir):
        os.makedirs(save_dict_dir)
    
    subj_data_folder = '/data1/xueyi/GRAB_processed_wsubj'
    clip_clean = np.load(seq_path)

    # print(f"clip_clean: {clip_clean.keys()}")

    pure_file_name = seq_path.split("/")[-1].split(".")[0]
    pure_subj_params_fn = f"{pure_file_name}_subj.npy"  
    
    subj_params_fn = os.path.join(subj_data_folder, split, pure_subj_params_fn)
    subj_params = np.load(subj_params_fn, allow_pickle=True).item()
    rhand_transl = subj_params["rhand_transl"]
    rhand_betas = subj_params["rhand_betas"]
    rhand_pose = clip_clean['f2']
    
    object_global_orient = clip_clean['f5'] ## clip_len x 3 --> orientation 
    object_trcansl = clip_clean['f6'] ## cliplen x 3 --> translation
    
    object_idx = clip_clean['f7'][0].item()
        
    # /data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes
    ## get the grab obj ###
    grab_path = "/data1/xueyi/GRAB_extracted"
    obj_mesh_path = os.path.join(grab_path, 'tools/object_meshes/contact_meshes')
    id2objmesh = []
    obj_meshes = sorted(os.listdir(obj_mesh_path))
    for i, fn in enumerate(obj_meshes):
        id2objmesh.append(os.path.join(obj_mesh_path, fn))
    cur_obj_mesh_fn = id2objmesh[object_idx]
    obj_mesh = trimesh.load(cur_obj_mesh_fn, process=False)
    obj_verts = np.array(obj_mesh.vertices)
    obj_vertex_normals = np.array(obj_mesh.vertex_normals)
    obj_faces = np.array(obj_mesh.faces)
    
    # save the object mesh with the pure file name . obj #
    cur_pure_mesh_sv_fn = f"{pure_file_name}_obj.obj"
    cur_pure_mesh_sv_fn = os.path.join(save_dict_dir, cur_pure_mesh_sv_fn)
    obj_mesh.export(cur_pure_mesh_sv_fn) # 
    
    cur_clip = (
        0, 60, clip_clean, 
        [clip_clean['f9'], clip_clean['f11'], clip_clean['f10'], clip_clean['f1'],  clip_clean['f2'], rhand_transl, rhand_betas, object_global_orient, object_trcansl, object_idx]
    )
    
    c = cur_clip
    
    # for i_data, cur_data in enumerate( c[3]):
    #     print(i_data, cur_data.shape) # classical
    
    object_id = c[3][-1]
    object_global_orient = c[3][-3] # num_frames x 3 
    object_transl = c[3][-2] # num_frames x 3 # nnframes 
    
    start_idx = 0
    window_size = 60 if window_size is None else window_size
    window_size = min(window_size, object_global_orient.shape[0])
    # print(f"object_global_orient: {object_global_orient.shape}, object_transl: {object_transl.shape}")
    object_global_orient = object_global_orient[start_idx: start_idx + window_size]
    object_transl = object_transl[start_idx: start_idx + window_size]
    
    object_global_orient = object_global_orient.reshape(window_size, -1).astype(np.float32)
    object_transl = object_transl.reshape(window_size, -1).astype(np.float32)
    
    # object_global_orient_mtx = batched_get_orientation_matrices(object_global_orient)
    # object_global_orient_mtx_th = torch.from_numpy(object_global_orient_mtx).float() ## global orientation matrix ##
    # object_trcansl_th = torch.from_numpy(object_transl).float()
    
    
    rhand_global_orient_gt, rhand_pose_gt = c[3][3], c[3][4]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
    rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + window_size]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
    rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + window_size]
    
    ## global 
    rhand_global_orient_gt = rhand_global_orient_gt.reshape(window_size, -1).astype(np.float32)
    rhand_pose_gt = rhand_pose_gt.reshape(window_size, -1).astype(np.float32)
    
    rhand_transl, rhand_betas = c[3][5], c[3][6]
    rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + window_size], rhand_betas
    
    # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
    rhand_transl = rhand_transl.reshape(window_size, -1).astype(np.float32)
    rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
    
    
    rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float()
    rhand_pose_var = torch.from_numpy(rhand_pose_gt).float()
    rhand_beta_var = torch.from_numpy(rhand_betas).float()
    rhand_transl_var = torch.from_numpy(rhand_transl).float()
        
    rhand_verts, rhand_joints = mano_layer(
        torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
        rhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), rhand_transl_var
    )
    ### rhand_joints: for joints ###
    rhand_verts = rhand_verts * 0.001
    rhand_joints = rhand_joints * 0.001

    data = c[2]
    
    object_pc = data['f3'][start_idx: start_idx + window_size].reshape(window_size, -1, 3).astype(np.float32)
    # if self.args.scale_obj > 1:
    #     object_pc = object_pc * self.args.scale_obj
    object_normal = data['f4'][start_idx: start_idx + window_size].reshape(window_size, -1, 3).astype(np.float32)
    object_pc_th = torch.from_numpy(object_pc).float() # num_frames x nn_obj_pts x 3 #
    object_normal_th = torch.from_numpy(object_normal).float() # nn_ogj x 3 #


    ## get the obejct orientation and ## for 
    sv_dict = {
        'rhand_global_orient_gt': rhand_global_orient_gt,
        'rhand_transl': rhand_transl * 0.001,
        'rhand_verts': rhand_verts.detach().cpu().numpy(),
        'object_pc': object_pc_th.detach().cpu().numpy(),
        'object_normal': object_normal_th.detach().cpu().numpy(),
        'object_global_orient': object_global_orient,  # .detach().cpu().numpy(),
        'object_transl': object_transl, # .detach().cpu().numpy(),
        'obj_faces': obj_faces,
        'obj_verts': obj_verts, # faces and vertes
        'obj_vertex_normals': obj_vertex_normals,
    }
    
    
    sv_dict_fn = os.path.join(save_dict_dir, f"{pure_file_name}_sv_dict_st_{start_idx}_ed_{start_idx + window_size}.npy")
    np.save(sv_dict_fn, sv_dict)
    # print(f"rhand_verts: {rhand_verts.shape}, object_pc: {object_pc.shape}, object_normal: {object_normal.shape}")
    print(f'saved to {sv_dict_fn}')


def get_scale_dyn_mano_w_kine_mano():
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/mano/mano_mean_nocoll_simplified.urdf"
    # mano_agent = dyn_model_act_mano_deformable.RobotAgent(xml_fn=model_path_mano) # robot #
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano) ## model path mano ## # 
    mano_agent = mano_agent
    # ''' Load the mano hand '''
    robo_hand_faces = mano_agent.robot_faces
    robo_mano_verts = mano_agent.robot_pts
    
    mano_path = "/data1/xueyi/mano_models/mano/models"
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=mano_path, # mano_root #
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    )
    
    rhand_verts, rhand_joints = mano_layer(
        torch.cat([torch.zeros((1, 3), dtype=torch.float32), torch.zeros((1, 24), dtype=torch.float32)], dim=-1),
        torch.randn((1, 10), dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)
    )
    ### rhand_joints: for joints ###
    rhand_verts = rhand_verts * 0.001
    rhand_joints = rhand_joints * 0.001
    
    rhand_verts = rhand_verts.detach().cpu()[0]
    
    robo_mano_verts = robo_mano_verts.detach().cpu()
    ## robo ##
    
    maxx_robo_mano, _ = torch.max(robo_mano_verts, dim=0)
    minn_robo_mano, _ = torch.min(robo_mano_verts, dim=0) ## 
    extent_robo_mano = maxx_robo_mano - minn_robo_mano
    extent_robo_mano = torch.sqrt(torch.sum(extent_robo_mano ** 2))
    
    maxx_rhand_verts, _ = torch.max(rhand_verts, dim=0)
    minn_rhand_verts, _ = torch.min(rhand_verts, dim=0) ##
    
    extent_rhand_verts = maxx_rhand_verts - minn_rhand_verts
    extent_rhand_verts = torch.sqrt(torch.sum(extent_rhand_verts ** 2))

    scale = extent_rhand_verts / extent_robo_mano
    print(scale)
    
    
def convert_ply_data_to_obj_data():
    ply_fn = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes/camera.ply"
    mesh = trimesh.load(ply_fn, process=False)
    obj_fn = "/data1/xueyi/GRAB_extracted/tools/object_meshes/contact_meshes/camera.obj"
    mesh.export(obj_fn)
    

if __name__=='__main__':
    
    # add the ground --- should be added according to the object ##
    # hand model ##
    # object model # 
    # the gound / plane model # -> then it is a simple scene for the mnaipulation task #
    # ---  train each scene with the world model # 
    # ---- get the optimized results #
    # retargeting datasets? #
    # from grab curating method #
    # carving the collision geometry for each object #
    
    
    # convert_ply_data_to_obj_data()
    # exit(0)
    
    # get_scale_dyn_mano_w_kine_mano() # 
    
    tot_nn_tests = 246
    tot_nn_tests = 1390
    st_idx = 0
    ed_idx = tot_nn_tests
    
    # selected_idx = 30
    
    # tst_seq_idx = 0
    # split = 'train'
    # test_seq_path = f"/data1/xueyi/GRAB_processed/{split}/{tst_seq_idx}.npy" # extract_hand_obj_data_tst
    # extract_hand_obj_data_tst(test_seq_path, split=split)
    # exit(0)
    
    # st_idx = selected_idx
    # ed_idx = selected_idx + 1
    
    window_size = 120
    
    ## seq idx in st_idx and ed_idx ##
    # for seq_idx in range(selected_idx, selected_idx + 1): # add the ground -- should be added according to the object ##
    for seq_idx in range(st_idx, ed_idx):
        
        split = "test" ## 
        split = 'train' 
        seq_path = f"/data1/xueyi/GRAB_processed/{split}/{seq_idx}.npy"
        try:
            extract_hand_obj_data(seq_path, split=split, window_size=window_size)
        except:
            pass
        
        ## grab --- a setting of grasping object from the table ##
        ## so we should better add a table for a more realistic setting ##
        
        ## does the network have enough capability to learn the residual physics between two settings? ##
        ## network design and networks for low level physics ##
        ## and the retargeting problem ---- we should change the morphology for a better test ## 
            ## 

