

import numpy as np
import pickle as pkl

import os 
import torch
import ControlVAECore.Model.dyn_model_act_v2 as dyn_model_act_mano

from scipy.spatial.transform import Rotation as R

import trimesh
from ControlVAECore.Model.dyn_model_act_v2 import load_active_passive_timestep_to_mesh_v3_taco

SUCC_ROT_DIFF_THRESHOLD = 0.1
SUCC_TRANS_DIFF_THRESHOLD = 0.05

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1) # -1 for the quaternion matrix # 
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    
    return o.reshape(quaternions.shape[:-1] + (3, 3))



def euler_state_to_quat_wxyz(euler_state):
    euler_state_np = euler_state.detach().cpu().numpy()
    euler_state_np = euler_state_np[[2, 1, 0]]
    euler_rot_struct = R.from_euler('zyx', euler_state_np, degrees=False)
    quat_xyzw = euler_rot_struct.as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    quat_wxyz = torch.tensor(quat_wxyz, dtype=torch.float32).cuda()
    return quat_wxyz
    


# tot_obj_rot_quat, object_transl , lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_ref_fn)
def load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_ref_fn):
    # train_dyn_mano_model_states ## rhand 
    from manopth.manolayer import ManoLayer
    # sv_fn = "/data1/xueyi/GRAB_extracted_test/test/30_sv_dict.npy"
    # /data1/xueyi/GRAB_extracted_test/train/20_sv_dict_real_obj.obj # data1
    # import utils.utils as utils
    # from manopth.manolayer import ManoLayer
    
    start_idx = 20 # start_idx 
    window_size = 150
    # start_idx = self.start_idx
    # window_size = self.window_size
    
    # if 'model.kinematic_mano_gt_sv_fn' in self.conf:
    # sv_fn = self.conf['model.kinematic_mano_gt_sv_fn']
    sv_fn = arctic_gt_ref_fn
    # 
    # gt_data_folder = "/".join(sv_fn.split("/")[:-1]) ## 
    # gt_data_fn_name = sv_fn.split("/")[-1].split(".")[0]
    # arctic_processed_data_sv_folder = "/root/diffsim/quasi-dyn/raw_data/arctic_processed_canon_obj"
    # gt_data_canon_obj_sv_fn = f"{arctic_processed_data_sv_folder}/{gt_data_fn_name}_canon_obj.obj"
        
    print(f"Loading data from {sv_fn}")
    
    sv_dict = np.load(sv_fn, allow_pickle=True).item()
    
    
    tot_frames_nn  = sv_dict["obj_rot"].shape[0] ## obj rot ##
    window_size = min(tot_frames_nn - start_idx, window_size)
    window_size = window_size
    
    
    object_global_orient = sv_dict["obj_rot"][start_idx: start_idx + window_size] # num_frames x 3 
    object_transl = sv_dict["obj_trans"][start_idx: start_idx + window_size] * 0.001 # num_frames x 3
    obj_pcs = sv_dict["verts.object"][start_idx: start_idx + window_size]
    
    # obj_pcs = sv_dict['object_pc']
    obj_pcs = torch.from_numpy(obj_pcs).float().cuda()
    
    
    obj_vertex_normals = torch.zeros_like(obj_pcs[0])
    obj_normals = obj_vertex_normals
    
    # /data/xueyi/sim/arctic_processed_data/processed_seqs/s01/espressomachine_use_01.npy
    
    # obj_vertex_normals = sv_dict['obj_vertex_normals']
    # obj_vertex_normals = torch.from_numpy(obj_vertex_normals).float().cuda()
    # self.obj_normals = obj_vertex_normals
    
    # object_global_orient = sv_dict['object_global_orient'] # glboal orient 
    # object_transl = sv_dict['object_transl']
    
    
    obj_faces = sv_dict['f'][0]
    obj_faces = torch.from_numpy(obj_faces).long().cuda()
    obj_faces = obj_faces
    
    # obj_verts = sv_dict['verts.object']
    # minn_verts = np.min(obj_verts, axis=0)
    # maxx_verts = np.max(obj_verts, axis=0)
    # extent = maxx_verts - minn_verts
    # # center_ori = (maxx_verts + minn_verts) / 2
    # # scale_ori = np.sqrt(np.sum(extent ** 2))
    # obj_verts = torch.from_numpy(obj_verts).float().cuda()
    
    
    # obj_sv_path = "/data3/datasets/xueyi/arctic/arctic_data/data/meta/object_vtemplates"
    # obj_name = sv_fn.split("/")[-1].split("_")[0]
    # obj_mesh_fn = os.path.join(obj_sv_path, obj_name, "mesh.obj")
    # print(f"loading from {obj_mesh_fn}")
    # # template_obj_vs, template_obj_fs = trimesh.load(obj_mesh_fn, force='mesh')
    # template_obj_vs, template_obj_fs = utils.read_obj_file_ours(obj_mesh_fn, sub_one=True)
    
    
    
    
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
    # nn_obj_verts x  3 ##
    ##### 
    # canon_obj_verts = torch.matmul(
    #     (init_obj_verts - init_obj_transl.unsqueeze(0)), init_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
    # )
    
    ''' canonical object verts '''
    # rot.T 
    canon_obj_verts = torch.matmul(
        init_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), (init_obj_verts - init_obj_transl.unsqueeze(0)).contiguous().transpose(1, 0).contiguous()
    ).contiguous().transpose(1, 0).contiguous()


    obj_verts = canon_obj_verts.clone()  
    # self.obj_verts = obj_verts.clone()  
    
    mesh_scale = 0.8
    bbmin, _ = obj_verts.min(0) #
    bbmax, _ = obj_verts.max(0) #
    
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # bounding box's max #
    # vertices = (vertices - center) * scale # (vertices - center) * scale # #
    
    
    tot_reversed_obj_rot_mtx = []
    tot_obj_quat = [] ## rotation matrix 
    
    re_transformed_obj_verts = []
    
    tot_obj_rot_quat = []
    
    # transformed_obj_verts = []
    for i_fr in range(object_global_orient.shape[0]):
        cur_glb_rot = object_global_orient[i_fr]
        cur_transl = object_transl[i_fr]
        cur_transl = torch.from_numpy(cur_transl).float().cuda()
        cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
        cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
        
        reversed_cur_glb_rot_mtx = cur_glb_rot_mtx.T
        reversed_cur_glb_rot_struct = R.from_matrix(reversed_cur_glb_rot_mtx)
        reversed_cur_glb_rot_quat = reversed_cur_glb_rot_struct.as_quat()
        reversed_cur_glb_rot_quat = reversed_cur_glb_rot_quat[[3, 0, 1, 2]]
        tot_obj_rot_quat.append(reversed_cur_glb_rot_quat)
        
        
        cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
        
        # transformed verts ## canon_verts x R + t = transformed_verts #
        # (transformed_verts - t) x R^T = canon_verts #
        # cur_transformed_verts = torch.matmul(
        #     self.obj_verts, cur_glb_rot_mtx
        # ) + cur_transl.unsqueeze(0)
        
        cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
        
        cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
        cur_obj_quat = cur_glb_rot_struct.as_quat()
        cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
        cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
        tot_obj_quat.append(cur_obj_quat)
        
        cur_re_transformed_obj_verts = torch.matmul(
            cur_glb_rot_mtx_reversed.transpose(1, 0).contiguous(), obj_verts.transpose(1, 0)
        ).transpose(1, 0) + cur_transl.unsqueeze(0)
        re_transformed_obj_verts.append(cur_re_transformed_obj_verts)


    
    transformed_obj_verts = obj_pcs.clone()
    
    
    tot_obj_rot_quat = np.stack(tot_obj_rot_quat, axis=0)
    
    # tot_obj_rot_quat, object_transl #
    # tot_obj_rot_quat = torch.from_numpy(tot_obj_rot_quat).float().cuda()   ### get the rot quat ---- the gtobj rotations daata
    # object_transl = torch.from_numpy(object_transl).float().cuda() ## get obj transl ### ## obj transl ##
    
    # mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
   
    
    if '30_sv_dict' in sv_fn:
        bbox_selected_verts_idxes = torch.tensor([1511, 1847, 2190, 2097, 2006, 2108, 1604], dtype=torch.long).cuda()
        obj_selected_verts = obj_verts[bbox_selected_verts_idxes]
    else:
        obj_selected_verts = obj_verts.clone()
    
    # maxx_init_passive_mesh, _ = torch.max(obj_selected_verts, dim=0)
    # minn_init_passive_mesh, _ = torch.min(obj_selected_verts, dim=0)
    # # self.maxx_init_passive_mesh = maxx_init_passive_mesh
    # self.minn_init_passive_mesh = minn_init_passive_mesh
    
    init_obj_verts = obj_verts
    
    mesh_scale = 0.8
    bbmin, _ = init_obj_verts.min(0) #
    bbmax, _ = init_obj_verts.max(0) #
    print(f"bbmin: {bbmin}, bbmax: {bbmax}")
    # center = (bbmin + bbmax) * 0.5
    
    
    mano_path = "/data1/xueyi/mano_models/mano/models" ### mano_path
    if not os.path.exists(mano_path):
        mano_path = '/data/xueyi/mano_v1_2/models'
    rgt_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='right',
        mano_root=mano_path,
        ncomps=45,
        use_pca=False,
    ).cuda()
    
    lft_mano_layer = ManoLayer(
        flat_hand_mean=False,
        side='left',
        mano_root=mano_path,
        ncomps=45,
        use_pca=False,
    ).cuda()
    
    
    ##### rhand parameters #####
    rhand_global_orient_gt, rhand_pose_gt = sv_dict["rot_r"], sv_dict["pose_r"]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
    rhand_global_orient_gt = rhand_global_orient_gt[start_idx: start_idx + window_size]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
    rhand_pose_gt = rhand_pose_gt[start_idx: start_idx + window_size]
    
    rhand_global_orient_gt = rhand_global_orient_gt.reshape(window_size, -1).astype(np.float32)
    rhand_pose_gt = rhand_pose_gt.reshape(window_size, -1).astype(np.float32)
    
    rhand_transl, rhand_betas = sv_dict["trans_r"], sv_dict["shape_r"][0]
    rhand_transl, rhand_betas = rhand_transl[start_idx: start_idx + window_size], rhand_betas
    
    # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
    rhand_transl = rhand_transl.reshape(window_size, -1).astype(np.float32)
    rhand_betas = rhand_betas.reshape(-1).astype(np.float32)
    
    rhand_global_orient_var = torch.from_numpy(rhand_global_orient_gt).float().cuda()
    rhand_pose_var = torch.from_numpy(rhand_pose_gt).float().cuda()
    rhand_beta_var = torch.from_numpy(rhand_betas).float().cuda()
    rhand_transl_var = torch.from_numpy(rhand_transl).float().cuda()
    # R.from_rotvec(obj_rot).as_matrix()
    ##### rhand parameters #####
    
    
    ##### lhand parameters #####
    lhand_global_orient_gt, lhand_pose_gt = sv_dict["rot_l"], sv_dict["pose_l"]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}")
    lhand_global_orient_gt = lhand_global_orient_gt[start_idx: start_idx + window_size]
    # print(f"rhand_global_orient_gt: {rhand_global_orient_gt.shape}, start_idx: {start_idx}, window_size: {self.window_size}, len: {self.len}")
    lhand_pose_gt = lhand_pose_gt[start_idx: start_idx + window_size]
    
    lhand_global_orient_gt = lhand_global_orient_gt.reshape(window_size, -1).astype(np.float32)
    lhand_pose_gt = lhand_pose_gt.reshape(window_size, -1).astype(np.float32)
    
    lhand_transl, lhand_betas = sv_dict["trans_l"], sv_dict["shape_l"][0]
    lhand_transl, lhand_betas = lhand_transl[start_idx: start_idx + window_size], lhand_betas
    
    # print(f"rhand_transl: {rhand_transl.shape}, rhand_betas: {rhand_betas.shape}")
    lhand_transl = lhand_transl.reshape(window_size, -1).astype(np.float32)
    lhand_betas = lhand_betas.reshape(-1).astype(np.float32)
    
    lhand_global_orient_var = torch.from_numpy(lhand_global_orient_gt).float().cuda()
    lhand_pose_var = torch.from_numpy(lhand_pose_gt).float().cuda()
    lhand_beta_var = torch.from_numpy(lhand_betas).float().cuda()
    lhand_transl_var = torch.from_numpy(lhand_transl).float().cuda() # self.window_size x 3
    # R.from_rotvec(obj_rot).as_matrix()
    ##### lhand parameters #####
    

    
    rhand_verts, rhand_joints = rgt_mano_layer(
        torch.cat([rhand_global_orient_var, rhand_pose_var], dim=-1),
        rhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), rhand_transl_var
    )
    ### rhand_joints: for joints ###
    rhand_verts = rhand_verts * 0.001
    rhand_joints = rhand_joints * 0.001
    
    
    lhand_verts, lhand_joints = lft_mano_layer(
        torch.cat([lhand_global_orient_var, lhand_pose_var], dim=-1),
        lhand_beta_var.unsqueeze(0).repeat(window_size, 1).view(-1, 10), lhand_transl_var
    )
    ### rhand_joints: for joints ###
    lhand_verts = lhand_verts * 0.001
    lhand_joints = lhand_joints * 0.001
    
    
    ### lhand and the rhand ###
    # rhand_verts, lhand_verts #
    # self.rhand_verts = rhand_verts
    # self.lhand_verts = lhand_verts 
    
    hand_faces = rgt_mano_layer.th_faces
    
    
    
    return tot_obj_rot_quat, object_transl , lhand_verts.detach().cpu().numpy(), rhand_verts.detach().cpu().numpy()
    return transformed_obj_verts, self.obj_normals




# tot_obj_quat, object_transl = load_active_passive_timestep_to_mesh_v3(kinematic_mano_gt_sv_fn)
def load_active_passive_timestep_to_mesh_v3(kinematic_mano_gt_sv_fn):

    print(f"Loading data from {kinematic_mano_gt_sv_fn}")
    
    sv_dict = np.load(kinematic_mano_gt_sv_fn, allow_pickle=True).item()
    
    print(f"sv_dict: {sv_dict.keys()}")
    
    # obj_pcs = sv_dict['object_pc']
    # obj_pcs = torch.from_numpy(obj_pcs).float().cuda()


    object_global_orient = sv_dict['object_global_orient'] # glboal orient 
    object_transl = sv_dict['object_transl']
    
    
    # tot_reversed_obj_rot_mtx = []
    tot_obj_quat = [] ## rotation matrix 
    
    
    # transformed_obj_verts = []
    for i_fr in range(object_global_orient.shape[0]):
        cur_glb_rot = object_global_orient[i_fr]
        cur_transl = object_transl[i_fr]
        cur_transl = torch.from_numpy(cur_transl).float().cuda()
        cur_glb_rot_struct = R.from_rotvec(cur_glb_rot)
        cur_glb_rot_mtx = cur_glb_rot_struct.as_matrix()
        cur_glb_rot_mtx = torch.from_numpy(cur_glb_rot_mtx).float().cuda()
        
        # cur_transformed_verts = torch.matmul(
        #     self.obj_verts, cur_glb_rot_mtx
        # ) + cur_transl.unsqueeze(0)
        
        cur_glb_rot_mtx_reversed = cur_glb_rot_mtx.contiguous().transpose(1, 0).contiguous()
        # tot_reversed_obj_rot_mtx.append(cur_glb_rot_mtx_reversed)
        
        cur_glb_rot_struct = R.from_matrix(cur_glb_rot_mtx_reversed.cpu().numpy())
        cur_obj_quat = cur_glb_rot_struct.as_quat()
        cur_obj_quat = cur_obj_quat[[3, 0, 1, 2]]
        cur_obj_quat = torch.from_numpy(cur_obj_quat).float().cuda()
        tot_obj_quat.append(cur_obj_quat)


    tot_obj_quat = torch.stack(tot_obj_quat, dim=0).detach().cpu().numpy()

    # object_transl = torch.from_numpy(object_transl).float().cuda()
    return tot_obj_quat, object_transl
        


    
def calc_obj_metrics(optimized_fn, gt_ref_fn, dict_key='state', obj_fn=None):
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    
    
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
        
        
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    if obj_fn is not None:
        obj_mesh = trimesh.load(obj_fn, force='mesh') ## load mesh
        obj_verts, obj_faces = obj_mesh.vertices, obj_mesh.faces ## vertices and faces ##
        transformed_gt_obj_verts = []
        transformed_optimized_obj_verts = []
    
    gt_ref_dict = np.load(gt_ref_fn, allow_pickle=True).item()
    gt_ref_obj_quat = gt_ref_dict['optimized_quat']
    gt_ref_obj_trans = gt_ref_dict['optimized_trans']
    
    optimized_obj_states = optimized_states[:, -7: ]
    optimized_obj_quat = optimized_obj_states[:, 3:]
    optimized_obj_trans = optimized_obj_states[:, :3]
    
    tot_rot_diff = []
    tot_trans_diff = []
    
    maxx_ws = 150
    ws = min(maxx_ws, optimized_obj_states.shape[0])
    
    # for i_fr in range(optimized_obj_states.shape[0]):
    for i_fr in range(ws):
        cur_gt_obj_quat = gt_ref_obj_quat[i_fr] # w xyz
        cur_opt_obj_quat = optimized_obj_quat[i_fr][[3, 0, 1, 2]] # w xyz
        cur_gt_obj_trans = gt_ref_obj_trans[i_fr]
        cur_opt_obj_trans = optimized_obj_trans[i_fr]
        
        diff_obj_quat = 1.0 - np.sum(cur_gt_obj_quat * cur_opt_obj_quat).item()
        diff_obj_trans = np.sqrt(np.sum((cur_gt_obj_trans - cur_opt_obj_trans) ** 2)).item()
        tot_rot_diff.append(diff_obj_quat)
        tot_trans_diff.append(diff_obj_trans)
        
        if obj_fn is not None:
            cur_gt_obj_rot_mtx = R.from_quat(cur_gt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_opt_obj_rot_mtx = R.from_quat(cur_opt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_gt_transformed_verts = np.matmul(cur_gt_obj_rot_mtx, obj_verts.T).T + cur_gt_obj_trans[None]
            cur_opt_transformed_verts = np.matmul(cur_opt_obj_rot_mtx, obj_verts.T).T + cur_opt_obj_trans[None]
            
            transformed_gt_obj_verts.append(cur_gt_transformed_verts)
            transformed_optimized_obj_verts.append(cur_opt_transformed_verts)


    avg_rot_diff = sum(tot_rot_diff) / float(len(tot_rot_diff))
    avg_trans_diff = sum(tot_trans_diff) / float(len(tot_trans_diff))
    final_rot_diff = tot_rot_diff[-1]
    final_trans_diff = tot_trans_diff[-1]
    
    avg_succ = (avg_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (avg_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    final_succ =  (final_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (final_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    
    if obj_fn is not None:
        transformed_gt_obj_verts = np.stack(transformed_gt_obj_verts, axis=0)
        transformed_optimized_obj_verts = np.stack(transformed_optimized_obj_verts, axis=0)
        return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts  
    
    return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ

# hand skeleton error #
# joint palm skeleton error #
### for the arctic hand tracking ##
def calcu_hand_tracking_errors(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
    ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
    
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
        
        
        
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
        
        
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    optimized_states = np.stack(optimized_states, axis=0)
    hand_optimized_states = optimized_states[:, :-7]
    # 
    trans_dim = 3
    rot_dim = 3
    rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 2]
    rhand_lhand_rot = hand_optimized_states[:, trans_dim * 2: (trans_dim + rot_dim) * 2]
    rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 2: ]
    
    hand_optimized_states = np.concatenate(
        [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    )
    
    optimized_states = hand_optimized_states
    
    
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    for i_fr in range(len(optimized_states)):
        cur_state = optimized_states[i_fr]
        cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        cur_trans = torch.from_numpy(cur_trans).float().cuda()
        cur_rot = torch.from_numpy(cur_rot).float().cuda()
        cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        rot_mtx = quaternion_to_matrix(cur_rot_quat)
        ## some problems ##
        mano_agent.set_init_states_target_value(cur_state)
        cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        ## visual hand pts ##
        cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        cur_rhand_verts_np, cur_hand_mesh_left_np, cur_transformed_obj = cur_gt_hand_obj_verts
        cur_gt_hand_pts = cur_rhand_verts_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts
    

### for the arctic hand tracking -- left hand ##
def calcu_hand_tracking_errors_lefthand(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    # model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left/shadowhand_left_new_scaled_nroot.urdf"
    # model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left/shadowhand_left_new_scaled_nnrot.urdf"
    if not os.path.exists(model_path_mano):
        # model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
        # model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description_left/shadowhand_left_new_scaled_nroot.urdf"
        model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description_left/shadowhand_left_new_scaled_nnrot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
    ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
    
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
        
        
        
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
        
        
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    optimized_states = np.stack(optimized_states, axis=0)
    hand_optimized_states = optimized_states[:, :-7]
    # 
    trans_dim = 3
    rot_dim = 3
    rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 2]
    rhand_lhand_rot = hand_optimized_states[:, trans_dim * 2: (trans_dim + rot_dim) * 2]
    rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 2: ]
    print(f"rhand_lhand_state: {rhand_lhand_state.shape}")
    
    hand_optimized_states = np.concatenate(
        [rhand_lhand_trans[:, 3:], rhand_lhand_rot[:, 3:], rhand_lhand_state[:, rhand_lhand_state.shape[-1] // 2: ]], axis=-1
    )
    
    
    optimized_states = hand_optimized_states
    
    
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    for i_fr in range(len(optimized_states)):
        cur_state = optimized_states[i_fr]
        cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        cur_trans = torch.from_numpy(cur_trans).float().cuda()
        cur_rot = torch.from_numpy(cur_rot).float().cuda()
        cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        rot_mtx = quaternion_to_matrix(cur_rot_quat)
        ## some problems ##
        cur_state = torch.cat(
            [torch.zeros((2,), dtype=torch.float32).cuda(), cur_state], dim=-1
        )
        mano_agent.set_init_states_target_value(cur_state)
        cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0)
        
        ## visual hand pts ##
        cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        cur_rhand_verts_np, cur_hand_mesh_left_np, cur_transformed_obj = cur_gt_hand_obj_verts
        cur_gt_hand_pts = cur_hand_mesh_left_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts
    


def calc_obj_metrics_taco(optimized_fn, gt_ref_fn, dict_key='state', obj_fn=None):
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    
    
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
        
        
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    if obj_fn is not None:
        obj_mesh = trimesh.load(obj_fn, force='mesh') ## load mesh
        obj_verts, obj_faces = obj_mesh.vertices, obj_mesh.faces ## vertices and faces ##
        transformed_gt_obj_verts = []
        transformed_optimized_obj_verts = []
    
    # gt_ref_dict = np.load(gt_ref_fn, allow_pickle=True).item()
    # gt_ref_obj_quat = gt_ref_dict['optimized_quat']
    # gt_ref_obj_trans = gt_ref_dict['optimized_trans']
    
    gt_ref_obj_quat, gt_ref_obj_trans = load_active_passive_timestep_to_mesh_v3_taco(gt_ref_fn)
    
    optimized_obj_states = optimized_states[:, -7: ]
    optimized_obj_quat = optimized_obj_states[:, 3:]
    optimized_obj_trans = optimized_obj_states[:, :3]
    
    tot_rot_diff = []
    tot_trans_diff = []
    
    tot_diff_obj_err = []
    
    maxx_ws = 150
    ws = min(maxx_ws, optimized_obj_states.shape[0])
    
    # for i_fr in range(optimized_obj_states.shape[0]):
    for i_fr in range(ws):
        cur_gt_obj_quat = gt_ref_obj_quat[i_fr] # w xyz
        cur_opt_obj_quat = optimized_obj_quat[i_fr][[3, 0, 1, 2]] # w xyz
        cur_gt_obj_trans = gt_ref_obj_trans[i_fr]
        cur_opt_obj_trans = optimized_obj_trans[i_fr]
        
        diff_obj_quat = 1.0 - np.sum(cur_gt_obj_quat * cur_opt_obj_quat).item()
        diff_obj_trans = np.sqrt(np.sum((cur_gt_obj_trans - cur_opt_obj_trans) ** 2)).item()
        tot_rot_diff.append(diff_obj_quat)
        tot_trans_diff.append(diff_obj_trans)
        
        if obj_fn is not None:
            cur_gt_obj_rot_mtx = R.from_quat(cur_gt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_opt_obj_rot_mtx = R.from_quat(cur_opt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_gt_transformed_verts = np.matmul(cur_gt_obj_rot_mtx, obj_verts.T).T + cur_gt_obj_trans[None]
            cur_opt_transformed_verts = np.matmul(cur_opt_obj_rot_mtx, obj_verts.T).T + cur_opt_obj_trans[None]
            
            transformed_gt_obj_verts.append(cur_gt_transformed_verts)
            transformed_optimized_obj_verts.append(cur_opt_transformed_verts)
            
            diff_transformed_obj_verts = np.sqrt(np.sum((cur_gt_transformed_verts - cur_opt_transformed_verts) ** 2, axis=-1))
            diff_transformed_obj_verts_avg = diff_transformed_obj_verts.mean()
            
            tot_diff_obj_err.append(diff_transformed_obj_verts_avg)




    avg_rot_diff = sum(tot_rot_diff) / float(len(tot_rot_diff))
    avg_trans_diff = sum(tot_trans_diff) / float(len(tot_trans_diff))
    final_rot_diff = tot_rot_diff[-1]
    final_trans_diff = tot_trans_diff[-1]
    
    avg_diff_obj_err = sum(tot_diff_obj_err) / float(len(tot_diff_obj_err))
    
    
    avg_succ = (avg_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (avg_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    final_succ =  (final_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (final_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    
    if obj_fn is not None:
        transformed_gt_obj_verts = np.stack(transformed_gt_obj_verts, axis=0)
        transformed_optimized_obj_verts = np.stack(transformed_optimized_obj_verts, axis=0)
        return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts  , avg_diff_obj_err
    
    return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, avg_diff_obj_err

    
### for the arctic hand tracking ##
def calcu_hand_tracking_errors_taco(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    
    if gt_data_fn.endswith(".pkl"):
        import pickle as pkl
        sv_dict = pkl.load(open(gt_data_fn, "rb"))
        ### the starting index is 40 here ###
        ts_to_hand_obj_verts = sv_dict['hand_verts'][40: ]
    else:
        gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
        ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
        ts_to_hand_obj_verts = {
            ts: ts_to_hand_obj_verts[ts][0] for ts in ts_to_hand_obj_verts
        }
    
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    optimized_states = np.stack(optimized_states, axis=0)
    hand_optimized_states = optimized_states[:, :-7]
    # 
    trans_dim = 3
    rot_dim = 3
    rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 1]
    rhand_lhand_rot = hand_optimized_states[:, trans_dim * 1: (trans_dim + rot_dim) * 1]
    rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 1: ]
    
    hand_optimized_states = np.concatenate(
        [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    )
    
    optimized_states = hand_optimized_states
    
    
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    for i_fr in range(len(optimized_states)):
        cur_state = optimized_states[i_fr]
        cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        cur_trans = torch.from_numpy(cur_trans).float().cuda()
        cur_rot = torch.from_numpy(cur_rot).float().cuda()
        cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        rot_mtx = quaternion_to_matrix(cur_rot_quat)
        ## some problems ##
        mano_agent.set_init_states_target_value(cur_state)
        cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        ## visual hand pts ##
        
        #### previous implementation ####
        # cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        # cur_rhand_verts_np, cur_transformed_obj = cur_gt_hand_obj_verts
        #### previous implementation ####
        
        #### ts to hand_verts only ####
        cur_rhand_verts_np = ts_to_hand_obj_verts[i_fr]
        #### ts to hand_verts only ####
        
        cur_gt_hand_pts = cur_rhand_verts_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts
    

### for the arctic hand tracking ##
def calcu_hand_tracking_errors_taco_v2(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    
    if gt_data_fn.endswith(".pkl"):
        import pickle as pkl
        sv_dict = pkl.load(open(gt_data_fn, "rb"))
        ### the starting index is 40 here ###
        ts_to_hand_obj_verts = sv_dict['hand_verts'][40: ]
    else:
        gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
        ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
        ts_to_hand_obj_verts = {
            ts: ts_to_hand_obj_verts[ts][0] for ts in ts_to_hand_obj_verts
        }
    
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    # saved_dict_fn
    ts_to_hand_obj_verts_opt = optimized_info_dict['ts_to_hand_obj_verts']
    
    
    mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, 86, 364, 477, 588, 699]
    robot_fingers = [6684, 9174, 53, 1623, 3209, 4495, 10028, 8762, 1030, 2266, 3822, 5058, 7074]
    mano_wrist_center_idx = mano_fingers[1]
    robot_wrist_center_idx = robot_fingers[1] ## 
    ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
    mano_fingers = np.array(mano_fingers, dtype=np.int32)
    robot_fingers = np.array(robot_fingers, dtype=np.int32) ##
    
    fingers_selected = [0, ] + list(range(2, len(mano_fingers)))
    fingers_selected = np.array(fingers_selected, dtype=np.int32)
    
    
    
    
    # if dict_key in optimized_info_dict:
    #     optimized_states = optimized_info_dict[dict_key]
    # elif dict_key == 'state_target':
    #     target = optimized_info_dict['target']
    #     state = optimized_info_dict['state']
    #     optimized_states  = [np.concatenate(
    #         [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
    #     ) for ii in range(len(state))]
    # else:
    #     raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    # optimized_states = np.stack(optimized_states, axis=0)
    # hand_optimized_states = optimized_states[:, :-7]
    # # 
    # trans_dim = 3
    # rot_dim = 3
    # rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 1]
    # rhand_lhand_rot = hand_optimized_states[:, trans_dim * 1: (trans_dim + rot_dim) * 1]
    # rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 1: ]
    
    # hand_optimized_states = np.concatenate(
    #     [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    # )
    
    # optimized_states = hand_optimized_states
    
    
    
    # if isinstance(optimized_states, list):
    #     optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    tot_joint_diff_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    tot_diff_robot_mano_fingers = []
    tot_sim_vector_wrist_to_fingers = []
    
    
    for i_fr in range(len(ts_to_hand_obj_verts_opt)):
        # cur_state = optimized_states[i_fr]
        # cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        # cur_trans = torch.from_numpy(cur_trans).float().cuda()
        # cur_rot = torch.from_numpy(cur_rot).float().cuda()
        # cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        # cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        # rot_mtx = quaternion_to_matrix(cur_rot_quat)
        # ## some problems ##
        # mano_agent.set_init_states_target_value(cur_state)
        # cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # # cur_visual_pts = cur_visual_pts * self.mult_after_center
        # cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        
        cur_visual_pts = ts_to_hand_obj_verts_opt[i_fr][0]
        
        cur_visual_pts = torch.from_numpy(cur_visual_pts).float().cuda()
        ## visual hand pts ##
        
        #### previous implementation ####
        # cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        # cur_rhand_verts_np, cur_transformed_obj = cur_gt_hand_obj_verts
        #### previous implementation ####
        
        #### ts to hand_verts only ####
        cur_rhand_verts_np = ts_to_hand_obj_verts[i_fr]
        #### ts to hand_verts only ####
        
        cur_gt_hand_pts = cur_rhand_verts_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        
        ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
        cur_robot_fingers = cur_visual_pts[robot_fingers]
        cur_mano_fingers = cur_gt_hand_pts[mano_fingers] ## mando fingers 
        cur_robot_wrist_center = cur_visual_pts[robot_wrist_center_idx]
        cur_mano_wrist_center = cur_gt_hand_pts[mano_wrist_center_idx] ##
        
        # print(f'cur_robot_fingers: {cur_robot_fingers.shape}, cur_mano_fingers: {cur_mano_fingers.shape}')
        
        # diff_robot_mano_fingers = np.sqrt(np.sum((cur_robot_fingers - cur_mano_fingers) ** 2, axis=-1)) # .mean()
        # diff_robot_mano_fingers = np.mean(diff_robot_mano_fingers)
        diff_robot_mano_fingers = torch.sqrt(torch.sum((cur_robot_fingers - cur_mano_fingers) ** 2, dim=-1)).mean().item()
        tot_diff_robot_mano_fingers.append(diff_robot_mano_fingers)
        ## diff wrist center to the xxx ## 
        
        ### robot fingers ###
        cur_robot_wrist_to_fingers = cur_robot_fingers[fingers_selected] - cur_robot_wrist_center[None]
        cur_mano_wrist_to_fingers = cur_mano_fingers[fingers_selected] - cur_mano_wrist_center[None] ## mano wrist centers
        # cur_robot_wrist_to_fingers 
        # cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / (np.sqrt(np.sum(cur_robot_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        # cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / (np.sqrt(np.sum(cur_mano_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        
        cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / torch.clamp(torch.norm(cur_robot_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / torch.clamp(torch.norm(cur_mano_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        
        # ## 
        # sim_vector_wrist_to_fingers = np.sum(
        #     cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, axis=-1 ## 
        # ).mean()
        
        sim_vector_wrist_to_fingers = torch.sum(
            cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, dim=-1 ## 
        ).mean().item() ## sim 
        tot_sim_vector_wrist_to_fingers.append(sim_vector_wrist_to_fingers)
        
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    
    avg_diff_robot_mano_fingers = sum(tot_diff_robot_mano_fingers) / float(len(tot_diff_robot_mano_fingers))
    avg_sim_vector_wrist_to_fingers = sum(tot_sim_vector_wrist_to_fingers) / float(len(tot_sim_vector_wrist_to_fingers))
    
    
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers
    


# angle_diff_quat, diff_trans, succ_d5_t5, succ_d10_t10, succ_d15_t15, succ_d20_t20 = calc_obj_tracking_metrics(gt_quat, gt_trans, obs_quat, obs_trans)

def calc_obj_tracking_metrics(gt_quat, gt_trans, obs_quat, obs_trans):
    dot_product_quat = np.sum(gt_quat * obs_quat, axis=-1) / (np.sqrt(np.sum(gt_quat ** 2, axis=-1)) * np.sqrt(np.sum(obs_quat ** 2, axis=-1)) + 1e-8)
    angle_diff_quat = np.arccos(dot_product_quat) * 180 / np.pi #
    # print(f"dot_product_quat: {dot_product_quat}, angle_diff_quat: {angle_diff_quat}")
    if np.isnan(angle_diff_quat):
        angle_diff_quat = 0.0
    diff_trans = np.sqrt(np.sum((gt_trans - obs_trans) ** 2)) ### 
    succ_d5_t5 = (angle_diff_quat <= 5.0) and (diff_trans <= 0.05)
    succ_d10_t10 = (angle_diff_quat <= 10.0) and (diff_trans <= 0.10)
    succ_d15_t15 = (angle_diff_quat <= 15.0) and (diff_trans <= 0.15)
    succ_d20_t20 = (angle_diff_quat <= 20.0) and (diff_trans <= 0.20)
    return angle_diff_quat, diff_trans, float(succ_d5_t5), float(succ_d10_t10), float(succ_d15_t15), float(succ_d20_t20)

## calculate metricso n the grab ##

def calc_obj_metrics_grab(optimized_fn, gt_ref_fn, dict_key='state', obj_fn=None):
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    
    
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
        
        
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
        
    print(f"optimized_states: {len(optimized_states)}")
    
    if obj_fn is not None:
        obj_mesh = trimesh.load(obj_fn, force='mesh') ## load mesh
        obj_verts, obj_faces = obj_mesh.vertices, obj_mesh.faces ## vertices and faces ##
        transformed_gt_obj_verts = []
        transformed_optimized_obj_verts = []
    
    # gt_ref_dict = np.load(gt_ref_fn, allow_pickle=True).item()
    # gt_ref_obj_quat = gt_ref_dict['optimized_quat']
    # gt_ref_obj_trans = gt_ref_dict['optimized_trans']
    
    gt_ref_obj_quat, gt_ref_obj_trans = load_active_passive_timestep_to_mesh_v3(gt_ref_fn)
    
    
    
    optimized_obj_states = optimized_states[:, -7: ]
    optimized_obj_quat = optimized_obj_states[:, 3:]
    optimized_obj_trans = optimized_obj_states[:, :3]
    
    tot_rot_diff = []
    tot_trans_diff = []
    
    tot_diff_obj_err = []
    
    maxx_ws = 150
    ws = min(maxx_ws, optimized_obj_states.shape[0])
    
    ## succ rate -> for ##
    ## 0.2 0
    
    ## 0.05
    # 1.0 - cos(quat1, quat2) ## rotat ## arccos()
    # np.arccos(dot_prod(quat1, quat2)) / np.pi * 180 --- 5 degree, 10 degree, 20 degree #
    # np.sqrt(np.sum((trans1 - trans2) ** 2)) --- 0.05, 0.1, 0.2 # 
    
    tot_angle_diff_quat, tot_diff_trans, tot_succ_d5_t5, tot_succ_d10_t10, tot_succ_d15_t15, tot_succ_d20_t20 = [], [], [], [], [], []
    
    # for i_fr in range(optimized_obj_states.shape[0]):
    for i_fr in range(ws):
        cur_gt_obj_quat = gt_ref_obj_quat[i_fr] # w xyz ## w xyz ##
        cur_opt_obj_quat = optimized_obj_quat[i_fr][[3, 0, 1, 2]] # w xyz
        cur_gt_obj_trans = gt_ref_obj_trans[i_fr]
        cur_opt_obj_trans = optimized_obj_trans[i_fr]
        
        diff_obj_quat = 1.0 - np.sum(cur_gt_obj_quat * cur_opt_obj_quat).item()
        diff_obj_trans = np.sqrt(np.sum((cur_gt_obj_trans - cur_opt_obj_trans) ** 2)).item()
        tot_rot_diff.append(diff_obj_quat)
        tot_trans_diff.append(diff_obj_trans)
        
        angle_diff_quat, diff_trans, succ_d5_t5, succ_d10_t10, succ_d15_t15, succ_d20_t20 = calc_obj_tracking_metrics(cur_gt_obj_quat, cur_gt_obj_trans, cur_opt_obj_quat, cur_opt_obj_trans)
        
        ## obj tracking succ ##
        tot_angle_diff_quat.append(angle_diff_quat)
        tot_diff_trans.append(diff_trans)
        tot_succ_d5_t5.append(succ_d5_t5)
        tot_succ_d10_t10.append(succ_d10_t10)
        tot_succ_d15_t15.append(succ_d15_t15)
        tot_succ_d20_t20.append(succ_d20_t20)
        
        if obj_fn is not None:
            cur_gt_obj_rot_mtx = R.from_quat(cur_gt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_opt_obj_rot_mtx = R.from_quat(cur_opt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_gt_transformed_verts = np.matmul(cur_gt_obj_rot_mtx, obj_verts.T).T + cur_gt_obj_trans[None]
            cur_opt_transformed_verts = np.matmul(cur_opt_obj_rot_mtx, obj_verts.T).T + cur_opt_obj_trans[None]
            
            transformed_gt_obj_verts.append(cur_gt_transformed_verts)
            transformed_optimized_obj_verts.append(cur_opt_transformed_verts)
            
            diff_transformed_obj_verts = np.sqrt(np.sum((cur_gt_transformed_verts - cur_opt_transformed_verts) ** 2, axis=-1))
            diff_transformed_obj_verts_avg = diff_transformed_obj_verts.mean()
            
            tot_diff_obj_err.append(diff_transformed_obj_verts_avg)


    avg_rot_diff = sum(tot_rot_diff) / float(len(tot_rot_diff))
    avg_trans_diff = sum(tot_trans_diff) / float(len(tot_trans_diff))
    final_rot_diff = tot_rot_diff[-1]
    final_trans_diff = tot_trans_diff[-1]
    
    avg_diff_obj_err = sum(tot_diff_obj_err) / float(len(tot_diff_obj_err))
    
    avg_succ = (avg_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (avg_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    final_succ =  (final_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (final_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    
    # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
    avg_angle_diff_quat = sum(tot_angle_diff_quat) / float(len(tot_angle_diff_quat))
    avg_diff_trans = sum(tot_diff_trans) / float(len(tot_diff_trans))
    avg_succ_d5_t5 = sum(tot_succ_d5_t5) / float(len(tot_succ_d5_t5))
    avg_succ_d10_t10 = sum(tot_succ_d10_t10) / float(len(tot_succ_d10_t10))
    avg_succ_d15_t15 = sum(tot_succ_d15_t15) / float(len(tot_succ_d15_t15))
    avg_succ_d20_t20 = sum(tot_succ_d20_t20) / float(len(tot_succ_d20_t20))
    
    if obj_fn is not None:
        transformed_gt_obj_verts = np.stack(transformed_gt_obj_verts, axis=0)
        transformed_optimized_obj_verts = np.stack(transformed_optimized_obj_verts, axis=0)
        return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans
    
    return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans



def calc_obj_metrics_arctic_v2(optimized_fn, gt_ref_data, dict_key='state', obj_fn=None):
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    
    
    ts_to_obj_quaternion = optimized_info_dict['ts_to_obj_quaternion']
    ts_to_obj_trans = optimized_info_dict['ts_to_obj_trans']
    tot_optimized_quat = []
    tot_optimized_trans = []
    for ts in range(len(ts_to_obj_quaternion)):
        cur_ts_obj_quat = ts_to_obj_quaternion[ts][[1, 2, 3, 0]]
        cur_ts_obj_trans = ts_to_obj_trans[ts]
        tot_optimized_quat.append(cur_ts_obj_quat)
        tot_optimized_trans.append(cur_ts_obj_trans)
    tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
    tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
    
    # if dict_key in optimized_info_dict:
    #     optimized_states = optimized_info_dict[dict_key]
    # elif dict_key == 'state_target':
    #     target = optimized_info_dict['target']
    #     state = optimized_info_dict['state']
    #     optimized_states  = [np.concatenate(
    #         [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
    #     ) for ii in range(len(state))]
        
        
    # else:
    #     raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    # if isinstance(optimized_states, list):
    #     optimized_states = np.stack(optimized_states, axis=0)
    
    if obj_fn is not None:
        obj_mesh = trimesh.load(obj_fn, force='mesh') ## load mesh
        obj_verts, obj_faces = obj_mesh.vertices, obj_mesh.faces ## vertices and faces ##
        transformed_gt_obj_verts = []
        transformed_optimized_obj_verts = []
    
    # gt_ref_dict = np.load(gt_ref_fn, allow_pickle=True).item()
    # gt_ref_obj_quat = gt_ref_dict['optimized_quat']
    # gt_ref_obj_trans = gt_ref_dict['optimized_trans']
    
    # gt_ref_obj_quat, gt_ref_obj_trans = load_active_passive_timestep_to_mesh_v3_taco(gt_ref_fn)
    
    
    gt_ref_obj_quat = gt_ref_data['obj_quat']
    gt_ref_obj_trans = gt_ref_data['obj_trans']
    
    
    # optimized_obj_states = optimized_states[:, -7: ]
    # optimized_obj_quat = optimized_obj_states[:, 3:]
    # optimized_obj_trans = optimized_obj_states[:, :3]
    
    optimized_obj_quat = tot_optimized_quat
    optimized_obj_trans = tot_optimized_trans
    
    
    tot_rot_diff = []
    tot_trans_diff = []
    
    tot_diff_obj_err = []
    
    maxx_ws = 150
    ws = min(maxx_ws, optimized_obj_quat.shape[0])
    
    tot_angle_diff_quat, tot_diff_trans, tot_succ_d5_t5, tot_succ_d10_t10, tot_succ_d15_t15, tot_succ_d20_t20 = [], [], [], [], [], []
    
    
    print(f"ws: {ws}")
    # for i_fr in range(optimized_obj_states.shape[0]):
    for i_fr in range(ws):
        cur_gt_obj_quat = gt_ref_obj_quat[i_fr] # w xyz
        cur_opt_obj_quat = optimized_obj_quat[i_fr][[3, 0, 1, 2]] # w xyz
        cur_gt_obj_trans = gt_ref_obj_trans[i_fr]
        cur_opt_obj_trans = optimized_obj_trans[i_fr]
        
        diff_obj_quat = 1.0 - np.sum(cur_gt_obj_quat * cur_opt_obj_quat).item()
        diff_obj_trans = np.sqrt(np.sum((cur_gt_obj_trans - cur_opt_obj_trans) ** 2)).item()
        tot_rot_diff.append(diff_obj_quat)
        tot_trans_diff.append(diff_obj_trans)
        
        angle_diff_quat, diff_trans, succ_d5_t5, succ_d10_t10, succ_d15_t15, succ_d20_t20 = calc_obj_tracking_metrics(cur_gt_obj_quat, cur_gt_obj_trans, cur_opt_obj_quat, cur_opt_obj_trans)
        
        
         ## obj tracking succ ##
        tot_angle_diff_quat.append(angle_diff_quat)
        tot_diff_trans.append(diff_trans)
        tot_succ_d5_t5.append(succ_d5_t5)
        tot_succ_d10_t10.append(succ_d10_t10)
        tot_succ_d15_t15.append(succ_d15_t15)
        tot_succ_d20_t20.append(succ_d20_t20)
        
        
        if obj_fn is not None:
            cur_gt_obj_rot_mtx = R.from_quat(cur_gt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_opt_obj_rot_mtx = R.from_quat(cur_opt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_gt_transformed_verts = np.matmul(cur_gt_obj_rot_mtx, obj_verts.T).T + cur_gt_obj_trans[None]
            cur_opt_transformed_verts = np.matmul(cur_opt_obj_rot_mtx, obj_verts.T).T + cur_opt_obj_trans[None]
            
            transformed_gt_obj_verts.append(cur_gt_transformed_verts)
            transformed_optimized_obj_verts.append(cur_opt_transformed_verts)
            
            diff_transformed_obj_verts = np.sqrt(np.sum((cur_gt_transformed_verts - cur_opt_transformed_verts) ** 2, axis=-1))
            diff_transformed_obj_verts_avg = diff_transformed_obj_verts.mean()
            
            tot_diff_obj_err.append(diff_transformed_obj_verts_avg)




    avg_rot_diff = sum(tot_rot_diff) / float(len(tot_rot_diff))
    avg_trans_diff = sum(tot_trans_diff) / float(len(tot_trans_diff))
    final_rot_diff = tot_rot_diff[-1]
    final_trans_diff = tot_trans_diff[-1]
    
    avg_diff_obj_err = sum(tot_diff_obj_err) / float(len(tot_diff_obj_err))
    
    
    avg_succ = (avg_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (avg_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    final_succ =  (final_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (final_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    
    
    # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
    avg_angle_diff_quat = sum(tot_angle_diff_quat) / float(len(tot_angle_diff_quat))
    avg_diff_trans = sum(tot_diff_trans) / float(len(tot_diff_trans))
    avg_succ_d5_t5 = sum(tot_succ_d5_t5) / float(len(tot_succ_d5_t5))
    avg_succ_d10_t10 = sum(tot_succ_d10_t10) / float(len(tot_succ_d10_t10))
    avg_succ_d15_t15 = sum(tot_succ_d15_t15) / float(len(tot_succ_d15_t15))
    avg_succ_d20_t20 = sum(tot_succ_d20_t20) / float(len(tot_succ_d20_t20))
    
    
    if obj_fn is not None:
        transformed_gt_obj_verts = np.stack(transformed_gt_obj_verts, axis=0)
        transformed_optimized_obj_verts = np.stack(transformed_optimized_obj_verts, axis=0)
        return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts  , avg_diff_obj_err,  avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans
    
    return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans



### for the arctic hand tracking ##
def calcu_hand_tracking_errors_arctic_v2(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    
    # if gt_data_fn.endswith(".pkl"):
    #     import pickle as pkl
    #     sv_dict = pkl.load(open(gt_data_fn, "rb"))
    #     ### the starting index is 40 here ###
    #     ts_to_hand_obj_verts = sv_dict['hand_verts'][40: ]
    # else:
    #     gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
    #     ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
    #     ts_to_hand_obj_verts = {
    #         ts: ts_to_hand_obj_verts[ts][0] for ts in ts_to_hand_obj_verts
    #     }
    
    ts_to_hand_obj_verts = gt_data_fn['rhand_verts']
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    # saved_dict_fn
    ts_to_hand_obj_verts_opt = optimized_info_dict['ts_to_hand_obj_verts']
    
    
    mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, 86, 364, 477, 588, 699]
    robot_fingers = [6684, 9174, 53, 1623, 3209, 4495, 10028, 8762, 1030, 2266, 3822, 5058, 7074]
    mano_wrist_center_idx = mano_fingers[1]
    robot_wrist_center_idx = robot_fingers[1] ## 
    ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
    mano_fingers = np.array(mano_fingers, dtype=np.int32)
    robot_fingers = np.array(robot_fingers, dtype=np.int32) ##
    
    fingers_selected = [0, ] + list(range(2, len(mano_fingers)))
    fingers_selected = np.array(fingers_selected, dtype=np.int32)
    
    
    
    
    # if dict_key in optimized_info_dict:
    #     optimized_states = optimized_info_dict[dict_key]
    # elif dict_key == 'state_target':
    #     target = optimized_info_dict['target']
    #     state = optimized_info_dict['state']
    #     optimized_states  = [np.concatenate(
    #         [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
    #     ) for ii in range(len(state))]
    # else:
    #     raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    # optimized_states = np.stack(optimized_states, axis=0)
    # hand_optimized_states = optimized_states[:, :-7]
    # # 
    # trans_dim = 3
    # rot_dim = 3
    # rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 1]
    # rhand_lhand_rot = hand_optimized_states[:, trans_dim * 1: (trans_dim + rot_dim) * 1]
    # rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 1: ]
    
    # hand_optimized_states = np.concatenate(
    #     [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    # )
    
    # optimized_states = hand_optimized_states
    
    
    
    # if isinstance(optimized_states, list):
    #     optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    tot_joint_diff_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    tot_diff_robot_mano_fingers = []
    tot_sim_vector_wrist_to_fingers = []
    
    
    for i_fr in range(len(ts_to_hand_obj_verts_opt)):
        # cur_state = optimized_states[i_fr]
        # cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        # cur_trans = torch.from_numpy(cur_trans).float().cuda()
        # cur_rot = torch.from_numpy(cur_rot).float().cuda()
        # cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        # cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        # rot_mtx = quaternion_to_matrix(cur_rot_quat)
        # ## some problems ##
        # mano_agent.set_init_states_target_value(cur_state)
        # cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # # cur_visual_pts = cur_visual_pts * self.mult_after_center
        # cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        
        cur_visual_pts = ts_to_hand_obj_verts_opt[i_fr][0]
        
        cur_visual_pts = torch.from_numpy(cur_visual_pts).float().cuda()
        ## visual hand pts ##
        
        #### previous implementation ####
        # cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        # cur_rhand_verts_np, cur_transformed_obj = cur_gt_hand_obj_verts
        #### previous implementation ####
        
        #### ts to hand_verts only ####
        cur_rhand_verts_np = ts_to_hand_obj_verts[i_fr]
        #### ts to hand_verts only ####
        
        cur_gt_hand_pts = cur_rhand_verts_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        
        ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
        cur_robot_fingers = cur_visual_pts[robot_fingers]
        cur_mano_fingers = cur_gt_hand_pts[mano_fingers] ## mando fingers 
        cur_robot_wrist_center = cur_visual_pts[robot_wrist_center_idx]
        cur_mano_wrist_center = cur_gt_hand_pts[mano_wrist_center_idx] ##
        
        print(f'cur_robot_fingers: {cur_robot_fingers.shape}, cur_mano_fingers: {cur_mano_fingers.shape}')
        
        # diff_robot_mano_fingers = np.sqrt(np.sum((cur_robot_fingers - cur_mano_fingers) ** 2, axis=-1)) # .mean()
        # diff_robot_mano_fingers = np.mean(diff_robot_mano_fingers)
        diff_robot_mano_fingers = torch.sqrt(torch.sum((cur_robot_fingers - cur_mano_fingers) ** 2, dim=-1)).mean().item()
        tot_diff_robot_mano_fingers.append(diff_robot_mano_fingers)
        ## diff wrist center to the xxx ## 
        
        ### robot fingers ###
        cur_robot_wrist_to_fingers = cur_robot_fingers[fingers_selected] - cur_robot_wrist_center[None]
        cur_mano_wrist_to_fingers = cur_mano_fingers[fingers_selected] - cur_mano_wrist_center[None] ## mano wrist centers
        # cur_robot_wrist_to_fingers 
        # cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / (np.sqrt(np.sum(cur_robot_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        # cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / (np.sqrt(np.sum(cur_mano_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        
        cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / torch.clamp(torch.norm(cur_robot_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / torch.clamp(torch.norm(cur_mano_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        
        # ## 
        # sim_vector_wrist_to_fingers = np.sum(
        #     cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, axis=-1 ## 
        # ).mean()
        
        sim_vector_wrist_to_fingers = torch.sum(
            cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, dim=-1 ## 
        ).mean().item() ## sim 
        tot_sim_vector_wrist_to_fingers.append(sim_vector_wrist_to_fingers)
        
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    
    avg_diff_robot_mano_fingers = sum(tot_diff_robot_mano_fingers) / float(len(tot_diff_robot_mano_fingers))
    avg_sim_vector_wrist_to_fingers = sum(tot_sim_vector_wrist_to_fingers) / float(len(tot_sim_vector_wrist_to_fingers))
    
    
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers


### for the arctic hand tracking ##
def calcu_hand_tracking_errors_arctic_v2_left(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    
    # if gt_data_fn.endswith(".pkl"):
    #     import pickle as pkl
    #     sv_dict = pkl.load(open(gt_data_fn, "rb"))
    #     ### the starting index is 40 here ###
    #     ts_to_hand_obj_verts = sv_dict['hand_verts'][40: ]
    # else:
    #     gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
    #     ts_to_hand_obj_verts = gt_data_dict['ts_to_hand_obj_verts']
    #     ts_to_hand_obj_verts = {
    #         ts: ts_to_hand_obj_verts[ts][0] for ts in ts_to_hand_obj_verts
    #     }
    
    ts_to_hand_obj_verts = gt_data_fn['lhand_verts']
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    # saved_dict_fn
    ts_to_hand_obj_verts_opt = optimized_info_dict['ts_to_hand_obj_verts']
    
    
    mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, 86, 364, 477, 588, 699]
    robot_fingers = [6684, 9174, 53, 1623, 3209, 4495, 10028, 8762, 1030, 2266, 3822, 5058, 7074]
    mano_wrist_center_idx = mano_fingers[1]
    robot_wrist_center_idx = robot_fingers[1] ## 
    ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
    mano_fingers = np.array(mano_fingers, dtype=np.int32)
    robot_fingers = np.array(robot_fingers, dtype=np.int32) ##
    
    fingers_selected = [0, ] + list(range(2, len(mano_fingers)))
    fingers_selected = np.array(fingers_selected, dtype=np.int32)
    
    
    
    
    # if dict_key in optimized_info_dict:
    #     optimized_states = optimized_info_dict[dict_key]
    # elif dict_key == 'state_target':
    #     target = optimized_info_dict['target']
    #     state = optimized_info_dict['state']
    #     optimized_states  = [np.concatenate(
    #         [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
    #     ) for ii in range(len(state))]
    # else:
    #     raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    # optimized_states = np.stack(optimized_states, axis=0)
    # hand_optimized_states = optimized_states[:, :-7]
    # # 
    # trans_dim = 3
    # rot_dim = 3
    # rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 1]
    # rhand_lhand_rot = hand_optimized_states[:, trans_dim * 1: (trans_dim + rot_dim) * 1]
    # rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 1: ]
    
    # hand_optimized_states = np.concatenate(
    #     [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    # )
    
    # optimized_states = hand_optimized_states
    
    
    
    # if isinstance(optimized_states, list):
    #     optimized_states = np.stack(optimized_states, axis=0)
    
    
    tot_cd_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    
    tot_joint_diff_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    tot_diff_robot_mano_fingers = []
    tot_sim_vector_wrist_to_fingers = []
    
    
    for i_fr in range(len(ts_to_hand_obj_verts_opt)):
        # cur_state = optimized_states[i_fr]
        # cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        # cur_trans = torch.from_numpy(cur_trans).float().cuda()
        # cur_rot = torch.from_numpy(cur_rot).float().cuda()
        # cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        # cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        # rot_mtx = quaternion_to_matrix(cur_rot_quat)
        # ## some problems ##
        # mano_agent.set_init_states_target_value(cur_state)
        # cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # # cur_visual_pts = cur_visual_pts * self.mult_after_center
        # cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        
        cur_visual_pts = ts_to_hand_obj_verts_opt[i_fr][1]
        
        cur_visual_pts = torch.from_numpy(cur_visual_pts).float().cuda()
        ## visual hand pts ##
        
        #### previous implementation ####
        # cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        # cur_rhand_verts_np, cur_transformed_obj = cur_gt_hand_obj_verts
        #### previous implementation ####
        
        #### ts to hand_verts only ####
        cur_rhand_verts_np = ts_to_hand_obj_verts[i_fr]
        #### ts to hand_verts only ####
        
        cur_gt_hand_pts = cur_rhand_verts_np
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        
        ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
        cur_robot_fingers = cur_visual_pts[robot_fingers]
        cur_mano_fingers = cur_gt_hand_pts[mano_fingers] ## mando fingers 
        cur_robot_wrist_center = cur_visual_pts[robot_wrist_center_idx]
        cur_mano_wrist_center = cur_gt_hand_pts[mano_wrist_center_idx] ##
        
        print(f'cur_robot_fingers: {cur_robot_fingers.shape}, cur_mano_fingers: {cur_mano_fingers.shape}')
        
        # diff_robot_mano_fingers = np.sqrt(np.sum((cur_robot_fingers - cur_mano_fingers) ** 2, axis=-1)) # .mean()
        # diff_robot_mano_fingers = np.mean(diff_robot_mano_fingers)
        diff_robot_mano_fingers = torch.sqrt(torch.sum((cur_robot_fingers - cur_mano_fingers) ** 2, dim=-1)).mean().item()
        tot_diff_robot_mano_fingers.append(diff_robot_mano_fingers)
        ## diff wrist center to the xxx ## 
        
        ### robot fingers ###
        cur_robot_wrist_to_fingers = cur_robot_fingers[fingers_selected] - cur_robot_wrist_center[None]
        cur_mano_wrist_to_fingers = cur_mano_fingers[fingers_selected] - cur_mano_wrist_center[None] ## mano wrist centers
        # cur_robot_wrist_to_fingers 
        # cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / (np.sqrt(np.sum(cur_robot_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        # cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / (np.sqrt(np.sum(cur_mano_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        
        cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / torch.clamp(torch.norm(cur_robot_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / torch.clamp(torch.norm(cur_mano_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        
        # ## 
        # sim_vector_wrist_to_fingers = np.sum(
        #     cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, axis=-1 ## 
        # ).mean()
        
        sim_vector_wrist_to_fingers = torch.sum(
            cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, dim=-1 ## 
        ).mean().item() ## sim 
        tot_sim_vector_wrist_to_fingers.append(sim_vector_wrist_to_fingers)
        
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    
    avg_diff_robot_mano_fingers = sum(tot_diff_robot_mano_fingers) / float(len(tot_diff_robot_mano_fingers))
    avg_sim_vector_wrist_to_fingers = sum(tot_sim_vector_wrist_to_fingers) / float(len(tot_sim_vector_wrist_to_fingers))
    
    
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers
     



### for the arctic hand tracking ##
def calcu_hand_tracking_errors_grab(optimized_fn, gt_data_fn, dict_key='state'):
    ##### shaodw hand ###
    model_path_mano = "/home/xueyi/diffsim/NeuS/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    if not os.path.exists(model_path_mano):
        model_path_mano = "/root/diffsim/quasi-dyn/rsc/shadow_hand_description/shadowhand_new_scaled_nroot.urdf"
    ## get the manojagent ##
    mano_agent = dyn_model_act_mano.RobotAgent(xml_fn=model_path_mano)
    # self.mano_agent  = mano_agent ## the mano agent for exporting meshes from the target simulator ### # simple world model ##
    
    

    
    gt_data_dict = np.load(gt_data_fn, allow_pickle=True).item()
    rhand_verts = gt_data_dict['rhand_verts']
    
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    # optimized_states = optimized_info_dict[dict_key]
    
    if dict_key in optimized_info_dict:
        optimized_states = optimized_info_dict[dict_key]
    elif dict_key == 'state_target':
        target = optimized_info_dict['target']
        state = optimized_info_dict['state']
        optimized_states  = [np.concatenate(
            [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
        ) for ii in range(len(state))]
    else:
        raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    ####### hand optimized states ########
    optimized_states = np.stack(optimized_states, axis=0)
    hand_optimized_states = optimized_states[:, :-7]
    # 
    trans_dim = 3
    rot_dim = 3
    rhand_lhand_trans = hand_optimized_states[:, :trans_dim * 1]
    rhand_lhand_rot = hand_optimized_states[:, trans_dim * 1: (trans_dim + rot_dim) * 1]
    rhand_lhand_state = hand_optimized_states[:, (trans_dim + rot_dim) * 1: ]
    
    hand_optimized_states = np.concatenate(
        [rhand_lhand_trans[:, :3], rhand_lhand_rot[:, :3], rhand_lhand_state], axis=-1
    )
    
    optimized_states = hand_optimized_states
    
    
    
    if isinstance(optimized_states, list):
        optimized_states = np.stack(optimized_states, axis=0)
    
    
    mano_fingers = [745, 279, 320, 444, 555, 672, 234, 121, 86, 364, 477, 588, 699]
    robot_fingers = [6684, 9174, 53, 1623, 3209, 4495, 10028, 8762, 1030, 2266, 3822, 5058, 7074]
    mano_wrist_center_idx = mano_fingers[1]
    robot_wrist_center_idx = robot_fingers[1] ## 
    ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
    mano_fingers = np.array(mano_fingers, dtype=np.int32)
    robot_fingers = np.array(robot_fingers, dtype=np.int32) ##
    
    fingers_selected = [0, ] + list(range(2, len(mano_fingers)))
    fingers_selected = np.array(fingers_selected, dtype=np.int32)
    
    
    
    tot_cd_loss = []
    
    tot_joint_diff_loss = []
    
    tot_visual_pts = []
    tot_gt_hand_pts = []
    tot_diff_robot_mano_fingers = []
    tot_sim_vector_wrist_to_fingers = []
    
    
    for i_fr in range(len(optimized_states)):
        cur_state = optimized_states[i_fr]
        cur_trans, cur_rot, cur_state = cur_state[:3], cur_state[3:6], cur_state[6:]
        
        cur_trans = torch.from_numpy(cur_trans).float().cuda()
        cur_rot = torch.from_numpy(cur_rot).float().cuda()
        cur_state = torch.from_numpy(cur_state).float().cuda()
        
        
        cur_rot_quat = euler_state_to_quat_wxyz(cur_rot)
        
        rot_mtx = quaternion_to_matrix(cur_rot_quat)
        ## some problems ##
        mano_agent.set_init_states_target_value(cur_state)
        cur_visual_pts = mano_agent.get_init_state_visual_pts()
        # cur_visual_pts = cur_visual_pts * self.mult_after_center
        cur_visual_pts = torch.matmul(rot_mtx, cur_visual_pts.contiguous().transpose(1, 0).contiguous()).transpose(1, 0).contiguous() + cur_trans.unsqueeze(0) # ### mano trans andjmano rot #3
        
        ## visual hand pts ##
        # cur_gt_hand_obj_verts = ts_to_hand_obj_verts[i_fr]
        # cur_rhand_verts_np, cur_transformed_obj = cur_gt_hand_obj_verts
        cur_gt_hand_pts = rhand_verts[i_fr]
        cur_gt_hand_pts = torch.from_numpy(cur_gt_hand_pts).float().cuda()
        
        ## finger tips #### pc cd loss ## ?
        dist_visual_pts_w_gt_hand_pts = torch.sum(
            (cur_visual_pts.unsqueeze(1) - cur_gt_hand_pts.unsqueeze(0)) ** 2, dim=-1
        )
        ## 
        dist_visual_pts_w_gt_hand_pts = torch.sqrt(dist_visual_pts_w_gt_hand_pts)
        dist_roto_to_mano, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=-1)
        dist_mano_to_robo, _ = torch.min(dist_visual_pts_w_gt_hand_pts, dim=0)
        
        dist_roto_to_mano = dist_roto_to_mano.mean()
        dist_mano_to_robo = dist_mano_to_robo.mean()
        
        cur_cd_loss = (dist_roto_to_mano + dist_mano_to_robo).item() / 2.
        tot_cd_loss.append(cur_cd_loss)
        
        ## mano_fingers, robot_fingers, mano_wrist_center_idx, robot_wrist_center_idx ##
        cur_robot_fingers = cur_visual_pts[robot_fingers]
        cur_mano_fingers = cur_gt_hand_pts[mano_fingers] ## mando fingers 
        cur_robot_wrist_center = cur_visual_pts[robot_wrist_center_idx]
        cur_mano_wrist_center = cur_gt_hand_pts[mano_wrist_center_idx] ##
        
        print(f'cur_robot_fingers: {cur_robot_fingers.shape}, cur_mano_fingers: {cur_mano_fingers.shape}')
        
        # diff_robot_mano_fingers = np.sqrt(np.sum((cur_robot_fingers - cur_mano_fingers) ** 2, axis=-1)) # .mean()
        # diff_robot_mano_fingers = np.mean(diff_robot_mano_fingers)
        diff_robot_mano_fingers = torch.sqrt(torch.sum((cur_robot_fingers - cur_mano_fingers) ** 2, dim=-1)).mean().item()
        tot_diff_robot_mano_fingers.append(diff_robot_mano_fingers)
        ## diff wrist center to the xxx ## 
        
        ### robot fingers ###
        cur_robot_wrist_to_fingers = cur_robot_fingers[fingers_selected] - cur_robot_wrist_center[None]
        cur_mano_wrist_to_fingers = cur_mano_fingers[fingers_selected] - cur_mano_wrist_center[None] ## mano wrist centers
        # cur_robot_wrist_to_fingers 
        # cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / (np.sqrt(np.sum(cur_robot_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        # cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / (np.sqrt(np.sum(cur_mano_wrist_to_fingers ** 2, axis=-1, keepdims=True)) + 1e-8)
        
        cur_robot_wrist_to_fingers = cur_robot_wrist_to_fingers / torch.clamp(torch.norm(cur_robot_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        cur_mano_wrist_to_fingers = cur_mano_wrist_to_fingers / torch.clamp(torch.norm(cur_mano_wrist_to_fingers, dim=-1, keepdim=True, p=2), min=1e-8)
        
        # ## 
        # sim_vector_wrist_to_fingers = np.sum(
        #     cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, axis=-1 ## 
        # ).mean()
        
        sim_vector_wrist_to_fingers = torch.sum(
            cur_robot_wrist_to_fingers * cur_mano_wrist_to_fingers, dim=-1 ## 
        ).mean().item() ## sim 
        tot_sim_vector_wrist_to_fingers.append(sim_vector_wrist_to_fingers)
        
        
        
        
        tot_visual_pts.append(cur_visual_pts.detach().cpu().numpy())
        tot_gt_hand_pts.append(cur_gt_hand_pts.detach().cpu().numpy())
        
    avg_cd_loss = sum(tot_cd_loss) / float(len(tot_cd_loss))
    
    avg_diff_robot_mano_fingers = sum(tot_diff_robot_mano_fingers) / float(len(tot_diff_robot_mano_fingers))
    avg_sim_vector_wrist_to_fingers = sum(tot_sim_vector_wrist_to_fingers) / float(len(tot_sim_vector_wrist_to_fingers))
    
    
    tot_visual_pts = np.stack(tot_visual_pts, axis=0)
    tot_gt_hand_pts = np.stack(tot_gt_hand_pts, axis=0)
    
    return avg_cd_loss, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers
    
    
def  test_taco_pkl_fn(taco_fn):
    sv_dict = pkl.load(open(taco_fn, "rb"))
    print(f"dict_keys: {sv_dict.keys()}")
    


def calc_obj_metrics_taco_v2(optimized_fn, gt_ref_fn, dict_key='state', obj_fn=None):
    optimized_info_dict = np.load(optimized_fn, allow_pickle=True).item()
    
    
    ts_to_obj_quaternion = optimized_info_dict['ts_to_obj_quaternion']
    ts_to_obj_trans = optimized_info_dict['ts_to_obj_trans']
    tot_optimized_quat = []
    tot_optimized_trans = []
    for ts in range(len(ts_to_obj_quaternion)):
        cur_ts_obj_quat = ts_to_obj_quaternion[ts][[1, 2, 3, 0]]
        cur_ts_obj_trans = ts_to_obj_trans[ts]
        tot_optimized_quat.append(cur_ts_obj_quat)
        tot_optimized_trans.append(cur_ts_obj_trans)
    tot_optimized_quat = np.stack(tot_optimized_quat, axis=0)
    tot_optimized_trans = np.stack(tot_optimized_trans, axis=0)
    
    # if dict_key in optimized_info_dict:
    #     optimized_states = optimized_info_dict[dict_key]
    # elif dict_key == 'state_target':
    #     target = optimized_info_dict['target']
    #     state = optimized_info_dict['state']
    #     optimized_states  = [np.concatenate(
    #         [state[ii][..., :-7], target[ii][..., -7:]], axis=-1
    #     ) for ii in range(len(state))]
        
        
    # else:
    #     raise ValueError(f"dict_key: {dict_key} not found in the optimized_info_dict")
    
    # if isinstance(optimized_states, list):
    #     optimized_states = np.stack(optimized_states, axis=0)
    
    if obj_fn is not None:
        obj_mesh = trimesh.load(obj_fn, force='mesh') ## load mesh
        obj_verts, obj_faces = obj_mesh.vertices, obj_mesh.faces ## vertices and faces ##
        transformed_gt_obj_verts = []
        transformed_optimized_obj_verts = []
    
    # gt_ref_dict = np.load(gt_ref_fn, allow_pickle=True).item()
    # gt_ref_obj_quat = gt_ref_dict['optimized_quat']
    # gt_ref_obj_trans = gt_ref_dict['optimized_trans']
    
    gt_ref_obj_quat, gt_ref_obj_trans = load_active_passive_timestep_to_mesh_v3_taco(gt_ref_fn)
    
    # optimized_obj_states = optimized_states[:, -7: ]
    # optimized_obj_quat = optimized_obj_states[:, 3:]
    # optimized_obj_trans = optimized_obj_states[:, :3]
    
    optimized_obj_quat = tot_optimized_quat
    optimized_obj_trans = tot_optimized_trans
    
    
    tot_rot_diff = []
    tot_trans_diff = []
    
    tot_diff_obj_err = []
    
    maxx_ws = 150
    ws = min(maxx_ws, optimized_obj_quat.shape[0])
    
    tot_angle_diff_quat, tot_diff_trans, tot_succ_d5_t5, tot_succ_d10_t10, tot_succ_d15_t15, tot_succ_d20_t20 = [], [], [], [], [], []
    
    
    print(f"ws: {ws}")
    # for i_fr in range(optimized_obj_states.shape[0]):
    for i_fr in range(ws):
        cur_gt_obj_quat = gt_ref_obj_quat[i_fr] # w xyz
        cur_opt_obj_quat = optimized_obj_quat[i_fr][[3, 0, 1, 2]] # w xyz
        cur_gt_obj_trans = gt_ref_obj_trans[i_fr]
        cur_opt_obj_trans = optimized_obj_trans[i_fr]
        
        diff_obj_quat = 1.0 - np.sum(cur_gt_obj_quat * cur_opt_obj_quat).item()
        diff_obj_trans = np.sqrt(np.sum((cur_gt_obj_trans - cur_opt_obj_trans) ** 2)).item()
        tot_rot_diff.append(diff_obj_quat)
        tot_trans_diff.append(diff_obj_trans)
        
        angle_diff_quat, diff_trans, succ_d5_t5, succ_d10_t10, succ_d15_t15, succ_d20_t20 = calc_obj_tracking_metrics(cur_gt_obj_quat, cur_gt_obj_trans, cur_opt_obj_quat, cur_opt_obj_trans)
        
        
         ## obj tracking succ ##
        tot_angle_diff_quat.append(angle_diff_quat)
        tot_diff_trans.append(diff_trans)
        tot_succ_d5_t5.append(succ_d5_t5)
        tot_succ_d10_t10.append(succ_d10_t10)
        tot_succ_d15_t15.append(succ_d15_t15)
        tot_succ_d20_t20.append(succ_d20_t20)
        
        
        if obj_fn is not None:
            cur_gt_obj_rot_mtx = R.from_quat(cur_gt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_opt_obj_rot_mtx = R.from_quat(cur_opt_obj_quat[[1, 2, 3, 0]]).as_matrix()
            cur_gt_transformed_verts = np.matmul(cur_gt_obj_rot_mtx, obj_verts.T).T + cur_gt_obj_trans[None]
            cur_opt_transformed_verts = np.matmul(cur_opt_obj_rot_mtx, obj_verts.T).T + cur_opt_obj_trans[None]
            
            transformed_gt_obj_verts.append(cur_gt_transformed_verts)
            transformed_optimized_obj_verts.append(cur_opt_transformed_verts)
            
            diff_transformed_obj_verts = np.sqrt(np.sum((cur_gt_transformed_verts - cur_opt_transformed_verts) ** 2, axis=-1))
            diff_transformed_obj_verts_avg = diff_transformed_obj_verts.mean()
            
            tot_diff_obj_err.append(diff_transformed_obj_verts_avg)




    avg_rot_diff = sum(tot_rot_diff) / float(len(tot_rot_diff))
    avg_trans_diff = sum(tot_trans_diff) / float(len(tot_trans_diff))
    final_rot_diff = tot_rot_diff[-1]
    final_trans_diff = tot_trans_diff[-1]
    
    avg_diff_obj_err = sum(tot_diff_obj_err) / float(len(tot_diff_obj_err))
    
    
    avg_succ = (avg_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (avg_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    final_succ =  (final_rot_diff <= SUCC_ROT_DIFF_THRESHOLD) and (final_trans_diff <= SUCC_TRANS_DIFF_THRESHOLD)
    
    
    # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
    avg_angle_diff_quat = sum(tot_angle_diff_quat) / float(len(tot_angle_diff_quat))
    avg_diff_trans = sum(tot_diff_trans) / float(len(tot_diff_trans))
    avg_succ_d5_t5 = sum(tot_succ_d5_t5) / float(len(tot_succ_d5_t5))
    avg_succ_d10_t10 = sum(tot_succ_d10_t10) / float(len(tot_succ_d10_t10))
    avg_succ_d15_t15 = sum(tot_succ_d15_t15) / float(len(tot_succ_d15_t15))
    avg_succ_d20_t20 = sum(tot_succ_d20_t20) / float(len(tot_succ_d20_t20))
    
    
    if obj_fn is not None:
        transformed_gt_obj_verts = np.stack(transformed_gt_obj_verts, axis=0)
        transformed_optimized_obj_verts = np.stack(transformed_optimized_obj_verts, axis=0)
        return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts  , avg_diff_obj_err,  avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans
    
    return avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20, tot_angle_diff_quat, tot_diff_trans




def eval_saved_dict(saved_dict_fn):
    saved_dict = np.load(saved_dict_fn, allow_pickle=True).item()
    print(f'keys: {saved_dict.keys()}')
    
    ts_to_hand_obj_verts = saved_dict['ts_to_hand_obj_verts']
    for ts in ts_to_hand_obj_verts:
        cur_hand_obj_verts = ts_to_hand_obj_verts[ts]
        print(type(cur_hand_obj_verts   ), len(cur_hand_obj_verts))


if __name__=='__main__':
    
    
    # saved_dict_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230923_001_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
    # eval_saved_dict(saved_dict_fn)
    # exit(0)
    
    
    DATASET = "arctic"
    # DATASET = "taco"
    DATASET = "grab"
    ## grab ## 
    
    # TACO_DATA_ROOT = "/data3/datasets/xueyi/taco/processed_data"
    # obj_idxx = "20230919_043"
    # obj_idxx_root = "20230919"
    
    # taco_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.pkl"
    # test_taco_pkl_fn(taco_fn=taco_fn)
    # exit(0)
    
    
    ctl_vae_root = "/root/diffsim/control-vae-2"
    if not os.path.exists(ctl_vae_root):
        ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
    
    GRAB_EXTRACTED_DATA_ROOT = "/data/xueyi/GRAB/GRAB_extracted_test/train"
    if not os.path.exists(GRAB_EXTRACTED_DATA_ROOT):
        GRAB_EXTRACTED_DATA_ROOT = "/data1/xueyi/GRAB_extracted_test/train"
    
    ## grab; arctic; ##
    
    if DATASET == 'grab':
        
        ###### sequences on the GRAB dataset ######
        obj_name = 'mouse'
        obj_idx = 102
        
        obj_name = 'hammer'
        obj_idx = 91
        
        # obj_name = 'stapler107'
        # obj_idx = 107
        
        # # obj_name = 'stapler107'
        # # obj_idx = 107
        
        # obj_name = 'flashlight'
        # obj_idx = 89
        
        # # obj_name = 'tiantianquan'
        # # obj_idx = 224
        
        # # obj_name = 'bunny'
        # # obj_idx = 85
        
        # obj_name = 'quadrangular'
        # obj_idx = 76
        
        # # obj_name = 'cylinder'
        # # obj_idx = 54
        
        # # obj_name = 'ball'
        # # obj_idx = 25
        
        # # obj_name = 'plane'
        # # obj_idx = 322
        
        # obj_name = 'train'
        # obj_idx = 398
        
        # # 
        
        # obj_name = 'mouse313'
        # obj_idx = 313
        
        # obj_name = 'hammer167'
        # obj_idx = 167
        
        # obj_name = 'watch'
        # obj_idx = 277
        
        # obj_name = 'bottle'
        # obj_idx = 296
        
        
        # obj_name = 'phone'
        # obj_idx = 110
        
        # obj_name = 'ball'
        # obj_idx = 25
        
        # obj_name = 'scissor'
        # obj_idx = 47
        
        # obj_name = 'cube'
        # obj_idx = 19
        
        # obj_name = 'handle'
        # obj_idx = 7
        
        # obj_name = 'cup'
        # obj_idx = 358
        
        obj_name = 'mouse'
        obj_idx = 102
        
        # obj_name = 'bunny'
        # obj_idx = 85
        
        # obj_name = 'flashlight'
        # obj_idx = 89

        # obj_name = 'stapler107'
        # obj_idx = 107
        
        # obj_name = 'hammer'
        # obj_idx = 91

        tag = 'wmwoana_ablations'
        
        dict_key = "state"
        
        gt_ref_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_sv_dict.npy") ## hammer? ##
        obj_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_obj.obj")
        
        
        ##### mouse - 102 #####
        optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_predmanoobj_woana_wtable_gn9d8_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_predmanoobj_woana_wtable_gn9d8__step"
        
        optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_predtarmano_wambient_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_predtarmano_wambient__step"
        
        optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_cs0d6_predtarmano_wambient_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_cs0d6_predtarmano_wambient__step"
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient__step"
        ##### mouse - 102 #####
        
        ##### bunny #####
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0__step"
        
        
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_3_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_3__step"
        ##### bunny #####
        
    
        
        
        ##### tiantianquan #####
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv5_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_1024_subiters_optim_params_shadow_224_tiantianquan_1_mas10000_tsv5__step"
        # ##### tiantianquan #####
        
        
        ##### quadrangular #####
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_76_quadrangular_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_76_quadrangular_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta__step"
        # ##### quadrangular #####
        
        
        # ##### cylinder #####
        # # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/1024_subiters_optim_params_shadow_54_cylinder_1_mas10000_"
        # # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_1024_subiters_optim_params_shadow_54_cylinder_1_mas10000__step"
        # ##### cylinder #####
        
        # ##### hammer #####
        # # /data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana_wtable_gn9d8_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana_wtable_gn9d8__step_10.npy
        # optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana_wtable_gn9d8_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana_wtable_gn9d8__step"
        
        
        # optimized_folder = "/root/diffsim/control-vae-2/ours_eval_files/grab/hammer_91"
        # # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma__step"
        
        # #### step ambientcnet step ###
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_std0d05_fromthres0d0_ambientcnet_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_std0d05_fromthres0d0_ambientcnet__step"
        ##### hammer #####
        
        # ##### stapler #####
        # optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_/"
        # optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar__step"
        
        # optimized_folder  = "/root/diffsim/control-vae-2/ours_eval_files/grab/stapler107_107"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3__step"
        
        # ##### isaac #####
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_fr100_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_fr100__step"
        # ##### isaac #####
        
        ##### stapler #####
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_v4_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_v4__step"
        
        ## isaac ##
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_4_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_4__step"
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_std0d01_netv1_mass10000_new_dp1d0_v1_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle_std0d01_netv1_mass10000_new_dp1d0_v1__step"
        
        #### isaac #####
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle_std0d01_netv1_mass10000_new_dp0d7_cnetpreddelta__step"
        #### isaac #####
        
        ####### phone #######
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd__step"
        
        ### isaac uses 1-10 ###
        
        ####### phone #######
        
        
        ########## ball #############
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball_adc_1_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball_adc_1__step"
        
        ### isaac ###
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_optim_params_25_ball__step"
        ### isaac ###
        
        ########## ball #############
        
        ########## scissors #############
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_47_scissor_std0d01_netv1_mass10000_wcnet_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_47_scissor_std0d01_netv1_mass10000_wcnet__step"
        
        optimized_folder= "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_47_scissor_std0d01_netv1_mass10000_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_47_scissor_std0d01_netv1_mass10000__step"
        ########## scissors #############
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_cube_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_obj_mass_1000_wregact_l101_l2001_splitact_sampletrajori_pl_19_trans10_nnorm_glbtar_ana_trajopt_wactnois_winits_v2_wmw0_lmv5_200_tsdiv_1000_kh_sv_subiter1024_objm1000_wm2_skdiv10_cube__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_7_handle_std0d01_netv1_mass7000_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_7_handle_std0d01_netv1_mass7000__step"
        
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_358_cup_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_obm10000_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_358_cup_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_obm10000__step"
        
        ##### flashlight #####
        # optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana__step"
        
        
        # optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana_policy_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana_policy__step"
        
        # ### isaac ###
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_1024_subiters_optim_params_shadow_89_flashlight_1_mas10000_b_wnl19_c_thres02_onres_d_net_k2_ntrj1024_ndelta__step"
        # ### isaac ###
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wocmase_wtable_gn_adp1d0_plr0d001_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wocmase_wtable_gn_adp1d0_plr0d001__step"
        ##### flashlight #####
        
        ##### flashlight #####
        
        ##### flashlight #####
        
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_322_plane_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_322_plane__step"
        
        
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_322_plane_trwmonly_tbv2_trpolicyonly_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_322_plane_trwmonly_tbv2_trpolicyonly__step"
        
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_398_train_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_398_train__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000_ntb_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000_ntb__step"
        
        
        # optimized_folder= "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000_ntb_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp0d3_trwmonly_cs0d6_predtarmano_wambient_obm15000_ntb__step"
        
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000__step"
        
        # optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_ldp10d0_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_ldp10d0__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana_policy_/"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana_policy__step"


        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana__step"


        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_isaac_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_isaac__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_softhres_0d3_"
        optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_softhres_0d3__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana__step"

        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_softhres_0d3_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_softhres_0d3__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_predmanoobj_woana__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle_predmanoobj_woana__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_predmanoobj_woana__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_predmanoobj_woana__step"
        
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_predmanoobj_woana__step"
        
        
        obj_name = 'flashlight'
        obj_idx = 89
        optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_predmanoobj_woana__step"


        # optimized_folder = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_predmanoobj_woana_"
        # optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_predmanoobj_woana__step"

        gt_ref_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_sv_dict.npy") ## hammer? ##
        obj_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_obj.obj") # 
        
        
        opt_st_idx = 0
        opt_ed_idx = 130 #  100
        opt_ed_idx = 30 #  100
        
        minn_avg_rot_diff = 1000
        minn_avg_trans_diff = 1000
        
        minn_final_rot_diff = 1000
        minn_final_trans_diff = 1000
        
        maxx_avg_succ = 0.0
        maxx_final_succ = 0.0
        
        minn_avg_diff_obj_err = 1000
        
        best_transformed_gt_obj_verts = None
        best_transformed_optimized_obj_verts = None
        
        minn_avg_obj_state_diff = 1000
        
        # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
        best_avg_angle_diff_quat = 1000.0
        best_avg_diff_trans = 1000.0
        best_avg_succ_d5_t5 = 0.0 # 
        best_avg_succ_d10_t10 = 0.0
        best_avg_succ_d15_t15 = 0.0
        best_avg_succ_d20_t20 = 0.0
        best_avg_diff_robot_mano_fingers = 1000.0
        best_avg_sim_vector_wrist_to_fingers = 0.0
        
        
        
        best_eval_idx = None
        
        # ########################## Metrics for a series of optimized files ##########################
        # for opt_idx in range(opt_st_idx, opt_ed_idx + 1):
        #     cur_opt_fn = os.path.join(optimized_folder, f"{optimized_fn_prefix}_{opt_idx}.npy") ## cur_opt_idx ##
        #     if not os.path.exists(cur_opt_fn):
        #         print(cur_opt_fn)
        #         continue
        #     # avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err   = calc_obj_metrics_grab(cur_opt_fn, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)
        #     avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20,  tot_angle_diff_quat, tot_diff_trans   = calc_obj_metrics_grab(cur_opt_fn, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)
            
        #     if avg_diff_obj_err < minn_avg_diff_obj_err:
        #         minn_avg_diff_obj_err = avg_diff_obj_err
        #         minn_avg_rot_diff = avg_rot_diff
        #         minn_avg_trans_diff = avg_trans_diff
                
        #         minn_final_rot_diff = final_rot_diff
        #         minn_final_trans_diff = final_trans_diff
                
                
        #         maxx_avg_succ = avg_succ
        #         maxx_final_succ = final_succ
                
        #         best_transformed_gt_obj_verts = transformed_gt_obj_verts
        #         best_transformed_optimized_obj_verts = transformed_optimized_obj_verts
                
        #         best_avg_angle_diff_quat = avg_angle_diff_quat
        #         best_avg_diff_trans = avg_diff_trans
        #         best_avg_succ_d5_t5 = avg_succ_d5_t5
        #         best_avg_succ_d10_t10 = avg_succ_d10_t10
        #         best_avg_succ_d15_t15 = avg_succ_d15_t15
        #         best_avg_succ_d20_t20 = avg_succ_d20_t20
                
            
        #         hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_grab(cur_opt_fn, gt_ref_fn,  dict_key=dict_key)
                
        #         best_eval_idx = opt_idx
                
        #         overall_succ_5d_5t_5ht = [tot_angle_diff_quat[i_fr] <= 5.0 and tot_diff_trans[i_fr] <= 0.05 and tot_diff_robot_mano_fingers[i_fr] <= 0.05 for i_fr in range(len(tot_angle_diff_quat))]
        #         overall_succ_10d_10t_10ht = [tot_angle_diff_quat[i_fr] <= 10.0 and tot_diff_trans[i_fr] <= 0.1 and tot_diff_robot_mano_fingers[i_fr] <= 0.1 for i_fr in range(len(tot_angle_diff_quat))]
        #         overall_succ_15d_15t_15ht = [tot_angle_diff_quat[i_fr] <= 15.0 and tot_diff_trans[i_fr] <= 0.15 and tot_diff_robot_mano_fingers[i_fr] <= 0.15 for i_fr in range(len(tot_angle_diff_quat))]
        #         overall_succ_20d_20t_20ht = [tot_angle_diff_quat[i_fr] <= 20.0 and tot_diff_trans[i_fr] <= 0.2 and tot_diff_robot_mano_fingers[i_fr] <= 0.2 for i_fr in range(len(tot_angle_diff_quat))]
                
        #         best_avg_diff_robot_mano_fingers = avg_diff_robot_mano_fingers
        #         best_avg_sim_vector_wrist_to_fingers = avg_sim_vector_wrist_to_fingers
        ########################## Metrics for a series of optimized files ##########################
        
        ########################## Metrics for the single optimized file ##########################
        ###### DGrasp-1 baseline #######
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85_tracking_2/2024-02-26-08-33-58/sv_info_500.npy"
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85_tracking_2/2024-02-26-08-33-58/sv_info_best.npy"
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer167_idx_167_tracking_2/2024-02-26-11-55-06/sv_info_best.npy"
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer167_idx_167_tracking_2/2024-02-26-11-55-06/sv_info_500.npy"
        dict_key = "tot_states"  ## ours sim ##
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_398_train_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_398_train_softhres_0d3__step_9.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_398_train_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_398_train__step_10.npy"
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313__step_3.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_313_mouse313_softhres_0d3_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_softhres_0d3__step_4.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse__step_4.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_softhres_0d3__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_softhres_0d3_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_softhres_0d3__step_0.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer__step_0.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107__step_0.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_softhres_0d3__step_1.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny__step_0.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_softhres_0d3__step_1.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_softhres_0d3__step_4.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight__step_4.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_76_quadrangular_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_76_quadrangular_softhres_0d3__step_1.npy"
        
        ## quadrangular ##
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_76_quadrangular_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_76_quadrangular__step_2.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167__step_10.npy"
        
        optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_softhres_0d3__step_0.npy"
        
        
        optimized_fn_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trwmonly_cs0d6_predtarmano_wambient__step_13.npy"
        
        optimized_fn_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2__step_1.npy"
        
        optimized_fn_info = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_wact_2__step_1.npy"
        
        
        
        dict_key = "state"
        
        obj_name = 'quadrangular'
        obj_idx = 76
        
        # obj_name = 'tiantianquan'
        # obj_idx = 224
        
        obj_name = 'flashlight'
        obj_idx = 89
        
        # obj_name = 'bottle'
        # obj_idx = 296
        
        # obj_name = 'train'
        # obj_idx = 398
        
        # obj_name = 'plane'
        # obj_idx = 322
        
        # obj_name = 'mouse313'
        # obj_idx = 313
        
        obj_name = 'hammer'
        obj_idx = 91
        
        # obj_name = 'watch'
        # obj_idx = 277
        
        # obj_name = 'phone'
        # obj_idx = 110
        
        # obj_name = 'mouse'
        # obj_idx = 102
        
        # obj_name = 'stapler107'
        # obj_idx = 107
        
        # obj_name = 'hammer167'
        # obj_idx = 167
        
        # obj_name = 'bunny'
        # obj_idx = 85

        tag = 'wmwoana_ablations'
        
        gt_ref_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_sv_dict.npy") ## hammer? ##
        obj_fn = os.path.join(GRAB_EXTRACTED_DATA_ROOT, f"{obj_idx}_obj.obj")
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_quadrangular_idx_76_tracking_2/2024-02-25-05-39-39/sv_info_best.npy"
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_quadrangular_idx_76_tracking_2/2024-02-25-05-39-39/sv_info_100.npy"
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_quadrangular_idx_76_tracking_2/2024-02-25-05-39-39/sv_info_best.npy"
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_quadrangular_idx_76_tracking_2/2024-02-25-05-39-39/sv_info_100.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_quadrangular_idx_76_tracking_2/2024-02-25-05-39-39/sv_info_1000.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_tiantianquan_idx_224_tracking_2/2024-02-25-09-24-31/sv_info_best.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85_tracking_2/2024-02-26-08-33-58/sv_info_800.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_flashlight_idx_89_tracking_2/2024-02-27-10-50-24/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_flashlight_idx_89_tracking_2/2024-02-27-10-50-24/sv_info_700.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer167_idx_167_tracking_2/2024-02-26-11-55-06/sv_info_800.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_train_idx_398_tracking_2/2024-02-28-06-58-22/sv_info_best.npy"
        # 
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_train_idx_398_tracking_2/2024-02-28-06-58-22/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_train_idx_398_tracking_2/2024-02-28-06-58-22/sv_info_700.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_plane_idx_322_tracking_2/2024-02-28-04-06-05/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_plane_idx_322_tracking_2/2024-02-28-04-06-05/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_plane_idx_322_tracking_2/2024-02-28-04-06-05/sv_info_700.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse313_idx_313_tracking_2/2024-02-26-06-56-55/sv_info_100.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse313_idx_313_tracking_2/2024-02-26-06-56-55/sv_info_best.npy"
        # # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse313_idx_313_tracking_2/2024-02-26-06-56-55/sv_info_1000.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer_idx_91_tracking_2/2024-02-27-15-06-22/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer_idx_91_tracking_2/2024-02-27-15-06-22/sv_info_1000.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse_idx_102_tracking_2/2024-02-27-05-44-09/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse_idx_102_tracking_2/2024-02-27-05-44-09/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse_idx_102_tracking_2/2024-02-27-05-44-09/sv_info_1000.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107_tracking_2/2024-02-28-03-27-09/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107_tracking_2/2024-02-28-03-27-09/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107_tracking_2/2024-02-28-03-27-09/sv_info_800.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bottle_idx_296_tracking_2/2024-02-28-15-29-43/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bottle_idx_296_tracking_2/2024-02-28-15-29-43/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bottle_idx_296_tracking_2/2024-02-28-15-29-43/sv_info_900.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_watch_idx_277_tracking_2/2024-02-28-14-15-29/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_watch_idx_277_tracking_2/2024-02-28-14-15-29/sv_info_500.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_watch_idx_277_tracking_2/2024-02-28-14-15-29/sv_info_900.npy"
        
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_phone_idx_110_tracking_2/2024-02-28-18-50-42/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_phone_idx_110_tracking_2/2024-02-28-18-50-42/sv_info_500.npy"
        
        ## tracking_2 ##
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_phone_idx_110_tracking_2/2024-02-28-18-50-42/sv_info_900.npy"
        
        
        
        dict_key = "tot_states"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85/2024-02-27-05-47-01/sv_info_best.npy"
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85/2024-02-27-05-47-01/sv_info_500.npy"
        
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_flashlight_idx_89/2024-02-28-07-36-34/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_flashlight_idx_89/2024-02-28-07-36-34/sv_info_500.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107/2024-02-29-16-48-38/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107/2024-02-29-16-48-38/sv_info_500.npy"
        
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer_idx_91/2024-02-28-18-12-22/sv_info_best.npy"
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer_idx_91/2024-02-28-18-12-22/sv_info_500.npy"
        
        dict_key = "tot_states"
        
        # dict_key = ""
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/tmp_ours/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_313_mouse313_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_wcmase_ni4_wtable_gn_adp1d0_trpolicyonly_cs0d6_predtarmano_wambient_wtb_mpc_v3__step_2.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_softhres_0d3_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_softhres_0d3__step_0.npy" 
        ## soften -- ##
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_softhres_0d3__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_softhres_0d3_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone_softhres_0d3__step_0.npy"
        # ## dict key ## ## state ##
        # dict_key = "state"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_isaac__step_0.npy"
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_296_bottle_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_296_bottle_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_110_phone_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_110_phone_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_76_quadrangular_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_76_quadrangular_isaac__step_0.npy"
        
        # optimized_fn_info ="/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_277_watch_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_277_watch_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_76_quadrangular_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_76_quadrangular_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_85_bunny_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_85_bunny_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_91_hammer_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_91_hammer_isaac__step_1.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_89_flashlight_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_89_flashlight_isaac__step_1.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_isaac_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_isaac__step_1.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_isaac_/evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_102_mouse_isaac__step_0.npy"
        
        # optimized_fn_info = "/data2/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_predmanoobj_woana__step_0_afcames.npy"
        
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_bunny_idx_85/2024-02-27-05-47-01/sv_info_best.npy"
        
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_mouse_idx_102_tracking_3/2024-03-12-05-49-46/sv_info_best.npy"
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_stapler107_idx_107_tracking_3/2024-03-12-05-54-16/sv_info_best.npy"
        optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_hammer_idx_91_tracking_3/2024-03-12-05-48-50/sv_info_best.npy"
        
        
        
        # dict_key = "state"
        
        
        tag = "dgrasp_1"
        
        ## metrics for the sinlge optimized file ##
        # ######################### Metrics for the single optimized file ##########################
        # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
        avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20,  tot_angle_diff_quat, tot_diff_trans  = calc_obj_metrics_grab(optimized_fn_info, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)

        
        if avg_diff_obj_err < minn_avg_diff_obj_err:
            minn_avg_diff_obj_err = avg_diff_obj_err
            minn_avg_rot_diff = avg_rot_diff
            minn_avg_trans_diff = avg_trans_diff
            
            minn_final_rot_diff = final_rot_diff
            minn_final_trans_diff = final_trans_diff
            
            
            maxx_avg_succ = avg_succ
            maxx_final_succ = final_succ
            
            best_transformed_gt_obj_verts = transformed_gt_obj_verts
            best_transformed_optimized_obj_verts = transformed_optimized_obj_verts
            
            best_avg_angle_diff_quat = avg_angle_diff_quat
            best_avg_diff_trans = avg_diff_trans
            best_avg_succ_d5_t5 = avg_succ_d5_t5
            best_avg_succ_d10_t10 = avg_succ_d10_t10
            best_avg_succ_d15_t15 = avg_succ_d15_t15
            best_avg_succ_d20_t20 = avg_succ_d20_t20
            
            
            # eval utils #
            # hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_arctic_v2(optimized_fn_info, gt_ref_fn,  dict_key=dict_key)
            
            
            hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_grab(optimized_fn_info, gt_ref_fn,  dict_key=dict_key)
            
            best_eval_idx = 0
        
            overall_succ_5d_5t_5ht = [tot_angle_diff_quat[i_fr] <= 5.0 and tot_diff_trans[i_fr] <= 0.05 and tot_diff_robot_mano_fingers[i_fr] <= 0.05 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_10d_10t_10ht = [tot_angle_diff_quat[i_fr] <= 10.0 and tot_diff_trans[i_fr] <= 0.1 and tot_diff_robot_mano_fingers[i_fr] <= 0.1 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_15d_15t_15ht = [tot_angle_diff_quat[i_fr] <= 15.0 and tot_diff_trans[i_fr] <= 0.15 and tot_diff_robot_mano_fingers[i_fr] <= 0.15 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_20d_20t_20ht = [tot_angle_diff_quat[i_fr] <= 20.0 and tot_diff_trans[i_fr] <= 0.2 and tot_diff_robot_mano_fingers[i_fr] <= 0.2 for i_fr in range(len(tot_angle_diff_quat))]
            
            best_avg_diff_robot_mano_fingers = avg_diff_robot_mano_fingers
            best_avg_sim_vector_wrist_to_fingers = avg_sim_vector_wrist_to_fingers
        # ######################### Metrics for the single optimized file ##########################
            
        ####### use the best object tracking result as the final result for calculation #########
        
        # optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_107_stapler107_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_fingerretar_confv3__step_4.npy"
        # optimized_folder = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_102_mouse_predmanoobj_woana_wtable_gn9d8_"
        # # gt_ref_fn = "/data1/xueyi/GRAB_extracted_test/train/107_sv_dict.npy"
        # # obj_fn = "/data1/xueyi/GRAB_extracted_test/train/107_obj.obj"
        # dict_key = "state"
        # avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts   = calc_obj_metrics_grab(optimized_fn, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)
        
        # hand_cd_err, tot_visual_pts, tot_gt_hand_pts = calcu_hand_tracking_errors_grab(optimized_fn, gt_ref_fn,  dict_key=dict_key)
        
        overall_succ_5d_5t_5ht = [float(val) for val in overall_succ_5d_5t_5ht]
        overall_succ_10d_10t_10ht = [float(val) for val in overall_succ_10d_10t_10ht]
        overall_succ_15d_15t_15ht = [float(val) for val in overall_succ_15d_15t_15ht]
        overall_succ_20d_20t_20ht = [float(val) for val in overall_succ_20d_20t_20ht]
        
        avg_succ_5d_5t_5ht = sum(overall_succ_5d_5t_5ht) / float(len(overall_succ_5d_5t_5ht))
        avg_succ_10d_10t_10ht = sum(overall_succ_10d_10t_10ht) / float(len(overall_succ_10d_10t_10ht))
        avg_succ_15d_15t_15ht = sum(overall_succ_15d_15t_15ht) / float(len(overall_succ_15d_15t_15ht))
        avg_succ_20d_20t_20ht  = sum(overall_succ_20d_20t_20ht) / float(len(overall_succ_20d_20t_20ht))
        
        
        eval_sv_dict = {
            'avg_rot_diff': minn_avg_rot_diff,
            'avg_trans_diff': minn_avg_trans_diff,
            'final_rot_diff': minn_final_rot_diff,
            'final_trans_diff': minn_final_trans_diff,
            'avg_succ': maxx_avg_succ,
            'final_succ': maxx_final_succ,
            'hand_cd_err': hand_cd_err,
            'transformed_gt_obj_verts': best_transformed_gt_obj_verts,
            'transformed_optimized_obj_verts': best_transformed_optimized_obj_verts,
            'tot_visual_pts': tot_visual_pts,
            'tot_gt_hand_pts': tot_gt_hand_pts,
        }
        
        save_eval_res_fn = os.path.join(ctl_vae_root, "eval_res", "grab")
        os.makedirs(save_eval_res_fn, exist_ok=True)
        
        save_eval_res_fn = os.path.join(save_eval_res_fn, obj_name)
        os.makedirs(save_eval_res_fn, exist_ok=True)
        
        sv_dict_fn = f"eval_sv_dict_grab_{obj_name}_tag_{tag}.npy"
        sv_dict_fn = os.path.join(save_eval_res_fn, sv_dict_fn)
        np.save(sv_dict_fn, eval_sv_dict)
        print(f"Evaluated res saved to {save_eval_res_fn}")
        
        print(f"Best eval idx: {best_eval_idx}, minn_avg_diff_obj_err: {minn_avg_diff_obj_err}, avg_rot_diff: {minn_avg_rot_diff}, avg_trans_diff: {minn_avg_trans_diff}, final_rot_diff: {minn_final_rot_diff}, final_trans_diff: {minn_final_trans_diff}, avg_succ: {maxx_avg_succ}, final_succ: {maxx_final_succ}")
        
        
        print(f"[Object] best_avg_angle_diff_quat: {best_avg_angle_diff_quat}, best_avg_diff_trans: {best_avg_diff_trans}, best_avg_succ_d5_t5: {best_avg_succ_d5_t5}, best_avg_succ_d10_t10: {best_avg_succ_d10_t10}, best_avg_succ_d15_t15: {best_avg_succ_d15_t15}, best_avg_succ_d20_t20: {best_avg_succ_d20_t20}")
        print(f"[Hand] best_avg_diff_robot_mano_fingers: {best_avg_diff_robot_mano_fingers}, best_avg_sim_vector_wrist_to_fingers: {best_avg_sim_vector_wrist_to_fingers}")
        print(f"[Hand & Object] avg_succ_5d_5t_5ht: {avg_succ_5d_5t_5ht}, avg_succ_10d_10t_10ht: {avg_succ_10d_10t_10ht}, avg_succ_15d_15t_15ht: {avg_succ_15d_15t_15ht}, avg_succ_20d_20t_20ht: {avg_succ_20d_20t_20ht}")
        
        succ_avg_5d_5t_5ht = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05 and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t_10ht = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t_15ht = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t_20ht = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 and best_avg_diff_robot_mano_fingers  <= 0.2
        
        succ_avg_5d_5t = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05  # and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 # and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 # and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 #  and best_avg_diff_robot_mano_fingers  <= 0.2
        
        print(f"[Hand Avg.] succ_avg_5d_5t: {succ_avg_5d_5t}, succ_avg_10d_10t: {succ_avg_10d_10t}, succ_avg_15d_15t: {succ_avg_15d_15t}, succ_avg_20d_20t: {succ_avg_20d_20t}")
        
        print(f"[Hand & Object Avg.] succ_avg_5d_5t_5ht: {succ_avg_5d_5t_5ht}, succ_avg_10d_10t_10ht: {succ_avg_10d_10t_10ht}, succ_avg_15d_15t_15ht: {succ_avg_15d_15t_15ht}, succ_avg_20d_20t_20ht: {succ_avg_20d_20t_20ht}")
        
        print(f"hand_cd_err: {hand_cd_err}")
    
    elif DATASET == "arctic":
        
        ### 
        dict_key = 'target'
        obj_name = "s01_phone"
        ctl_vae_root = "/root/diffsim/control-vae-2"
        optimized_fn = "/data/xueyi/control-vae/exp/subiters1024_optim_params_shadow_phone__mpc_v3_whandloss_wanbient_optobj_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_phone__mpc_v3_whandloss_wanbient_optobj__step_0.npy"
        gt_ref_fn = "/root/diffsim/control-vae-2/Data/ReferenceData/shadow_arctic_phone_data_st_20_optobj.npy"
        obj_fn = "/root/diffsim/control-vae-2/assets/arctic/phone.obj"
        
        
        dict_key = 'target'
        obj_name = "s01_scissors"
        # ctl_vae_root = "/root/diffsim/control-vae-2"
        ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
        optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_scissors__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_scissors__mpc_v3_whandloss__step_0.npy"
        gt_ref_fn = f"{ctl_vae_root}/Data/ReferenceData/shadow_arctic_scissors_data_st_20_optobj.npy" ## ref fn ->  with states information ##
        obj_fn = f"{ctl_vae_root}/assets/arctic/scissors.obj"
        
        
        ## target and the targe ### ## targets ##
        
        # dict_key = 'target'
        # obj_name = "s01_scissors"
        # # ctl_vae_root = "/root/diffsim/control-vae-2"
        # ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
        # optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_scissors__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_scissors__mpc_v3_whandloss__step_0.npy"
        # gt_ref_fn = f"{ctl_vae_root}/Data/ReferenceData/shadow_arctic_scissors_data_st_20_optobj.npy" ## ref fn ->  with states information ##
        # obj_fn = f"{ctl_vae_root}/assets/arctic/scissors.obj"
        
        
        opt_st_idx = 0
        opt_ed_idx = 100
        
        minn_avg_rot_diff = 1000
        minn_avg_trans_diff = 1000
        
        minn_final_rot_diff = 1000
        minn_final_trans_diff = 1000
        
        maxx_avg_succ = 0.0
        maxx_final_succ = 0.0
        
        minn_avg_diff_obj_err = 1000
        
        best_transformed_gt_obj_verts = None
        best_transformed_optimized_obj_verts = None
        
        minn_avg_obj_state_diff = 1000
        
        # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
        best_avg_angle_diff_quat = 1000.0
        best_avg_diff_trans = 1000.0
        best_avg_succ_d5_t5 = 0.0 # 
        best_avg_succ_d10_t10 = 0.0
        best_avg_succ_d15_t15 = 0.0
        best_avg_succ_d20_t20 = 0.0
        best_avg_diff_robot_mano_fingers = 1000.0
        best_avg_sim_vector_wrist_to_fingers = 0.0
        
        
        
        
        subj_obj_name = "s04_notebook"
        optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_notebook_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_notebook_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
        
        
        
        # subj_obj_name = "s02_phone"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_phone_optmanorules_softv0_thres0d3_nres_/meshes/00581249.npy"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_phone_optmanorules_softv0_thres0d3_nres_/meshes/00590189.npy"
        
        
        # subj_obj_name = "s04_capsulemachine"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_capsulemachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_/meshes/00000894.npy"
        
        
        # subj_obj_name = "s05_espressomachine"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_espressomachine_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
        
        
        # subj_obj_name = "s05_ketchup"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s05_ketchup_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00000000.npy"
        
        
        # subj_obj_name = "s02_ketchup"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_ketchup_optmanorules_softv0_thres0d3_nres_optrobo_thres0d2_/meshes/00207557.npy"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s02_ketchup_optmanorules_softv0_thres0d3_nres_optrobo_thres0d2_optrobo_thres0d1_/meshes/00137229.npy"
        
        # # subj_obj_name = "s06_microwave"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_microwave_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_/meshes/00000298.npy"
        
        
        # subj_obj_name = "s06_mixer"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_mixer_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_mixer_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
        
        
        # subj_obj_name = "s06_scissors"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_scissors_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00003129.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s06_scissors_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000149.npy"
        
        # subj_obj_name = "s04_box"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_box_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00003129.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_s04_box_optmanorules_softv0_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_/meshes/00000298.npy"

        
        # subj_obj_name = "s01_laptop"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_laptop_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/meshes/00450129.npy"
        # optimized_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_laptop_optmanorules_softv0_thres0d2_optdp_optrulesfrrobo_alltks_/meshes/00396489.npy"
        
        
        subj_idx = subj_obj_name.split("_")[0]
        obj_name = subj_obj_name.split("_")[1]
        
        obj_fn =  f"/home/xueyi/diffsim/NeuS/raw_data/arctic_processed_canon_obj/{subj_obj_name}.obj"
        
        arctic_gt_data_fn = f"/data/xueyi/sim/arctic_processed_data/processed_seqs/{subj_idx}/{obj_name}_grab_01.npy" 
        
        tot_obj_rot_quat, object_transl , lhand_verts, rhand_verts = load_active_passive_timestep_to_mesh_twohands_arctic(arctic_gt_data_fn) 
        
        
        
        gt_ref_data = {
            'obj_trans': object_transl, 
            'obj_quat': tot_obj_rot_quat
        }
        
        
        avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20,  tot_angle_diff_quat, tot_diff_trans  = calc_obj_metrics_arctic_v2(optimized_fn, gt_ref_data, dict_key='state', obj_fn=obj_fn)
        
        gt_ref_data_hand = {
            'rhand_verts': rhand_verts,
            'lhand_verts': lhand_verts
        }
        
        
        if avg_diff_obj_err < minn_avg_diff_obj_err:
            minn_avg_diff_obj_err = avg_diff_obj_err
            minn_avg_rot_diff = avg_rot_diff
            minn_avg_trans_diff = avg_trans_diff
            
            minn_final_rot_diff = final_rot_diff
            minn_final_trans_diff = final_trans_diff
            
            
            maxx_avg_succ = avg_succ
            maxx_final_succ = final_succ
            
            best_transformed_gt_obj_verts = transformed_gt_obj_verts
            best_transformed_optimized_obj_verts = transformed_optimized_obj_verts
            
            best_avg_angle_diff_quat = avg_angle_diff_quat
            best_avg_diff_trans = avg_diff_trans
            best_avg_succ_d5_t5 = avg_succ_d5_t5
            best_avg_succ_d10_t10 = avg_succ_d10_t10
            best_avg_succ_d15_t15 = avg_succ_d15_t15
            best_avg_succ_d20_t20 = avg_succ_d20_t20
            
            
            hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_arctic_v2(optimized_fn, gt_ref_data_hand,  dict_key=dict_key)
            
            
            hand_cd_err_left, tot_visual_pts_left, tot_gt_hand_pts, avg_diff_robot_mano_fingers_left, avg_sim_vector_wrist_to_fingers_left, tot_diff_robot_mano_fingers_left, tot_sim_vector_wrist_to_fingers_left = calcu_hand_tracking_errors_arctic_v2(optimized_fn, gt_ref_data_hand,  dict_key=dict_key)
            
            hand_cd_err = (hand_cd_err + hand_cd_err_left) / 2.0
            avg_diff_robot_mano_fingers = (avg_diff_robot_mano_fingers + avg_diff_robot_mano_fingers_left) / 2.0
            avg_sim_vector_wrist_to_fingers = (avg_sim_vector_wrist_to_fingers + avg_sim_vector_wrist_to_fingers_left) / 2.0
            tot_diff_robot_mano_fingers = [ (cur_diff + cur_diff_left) / 2.0 for cur_diff, cur_diff_left in zip(tot_diff_robot_mano_fingers, tot_diff_robot_mano_fingers_left) ]
            tot_sim_vector_wrist_to_fingers = [ (cur_sim + cur_sim_left) / 2.0 for cur_sim, cur_sim_left in zip(tot_sim_vector_wrist_to_fingers, tot_sim_vector_wrist_to_fingers_left)]
            
            
            
            # hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_grab(optimized_fn_info, gt_ref_fn,  dict_key=dict_key)
            
            best_eval_idx = 0
        
            overall_succ_5d_5t_5ht = [tot_angle_diff_quat[i_fr] <= 5.0 and tot_diff_trans[i_fr] <= 0.05 and tot_diff_robot_mano_fingers[i_fr] <= 0.05 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_10d_10t_10ht = [tot_angle_diff_quat[i_fr] <= 10.0 and tot_diff_trans[i_fr] <= 0.1 and tot_diff_robot_mano_fingers[i_fr] <= 0.1 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_15d_15t_15ht = [tot_angle_diff_quat[i_fr] <= 15.0 and tot_diff_trans[i_fr] <= 0.15 and tot_diff_robot_mano_fingers[i_fr] <= 0.15 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_20d_20t_20ht = [tot_angle_diff_quat[i_fr] <= 20.0 and tot_diff_trans[i_fr] <= 0.2 and tot_diff_robot_mano_fingers[i_fr] <= 0.2 for i_fr in range(len(tot_angle_diff_quat))]
            
            best_avg_diff_robot_mano_fingers = avg_diff_robot_mano_fingers
            best_avg_sim_vector_wrist_to_fingers = avg_sim_vector_wrist_to_fingers
        ######################### Metrics for the single optimized file ##########################
            
        
        
        
        
        
        
        
        
        
        ##################### original implementation #####################
        # optimized_fn_info = "/home/xueyi/diffsim/raisim/dgrasp/raisimGymTorch/bullet_env/obj_flashlight_idx_89_tracking_2/2024-02-27-10-50-24/sv_info_500.npy"
        # dict_key = "tot_states"
        
   
        # avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20,  tot_angle_diff_quat, tot_diff_trans  = calc_obj_metrics_grab(optimized_fn_info, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)

        # avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts   = calc_obj_metrics(optimized_fn, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)
    
        # save_eval_res_fn = os.path.join(ctl_vae_root, "eval_res", "arctic", obj_name)
        # os.makedirs(save_eval_res_fn, exist_ok=True)
        
        # print(f"avg_rot_diff: {avg_rot_diff}, avg_trans_diff: {avg_trans_diff}, final_rot_diff: {final_rot_diff}, final_trans_diff: {final_trans_diff}, avg_succ: {avg_succ}, final_succ: {final_succ}")
        
        # ### gt_data_fn -> for rhand verts pts ##
        # gt_data_fn = "/data/xueyi/NeuS/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_phone_optmanorules_wresum_/meshes/00173719.npy"
        
        # gt_data_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_scissors_optmanorules_softv0_thres0d2_adp1d0_/meshes/01415649.npy"
        
        # # obj_states_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__arctic_scissors_optmanorules_softv0_thres0d2_adp1d0_/meshes/01415649.npy"
        
        # ## hand cd err ##
        # ### tot visual pts ###
        # ### to gt hand pts ###
        
        # hand_cd_err, tot_visual_pts, tot_gt_hand_pts = calcu_hand_tracking_errors(optimized_fn, gt_data_fn,  dict_key=dict_key)
        
        # left_hand_cd_err, left_tot_visual_pts, left_tot_gt_hand_pts = calcu_hand_tracking_errors_lefthand(optimized_fn, gt_data_fn,  dict_key=dict_key)
        
        ##################### original implementation #####################
        
        
        
        overall_succ_5d_5t_5ht = [float(val) for val in overall_succ_5d_5t_5ht]
        overall_succ_10d_10t_10ht = [float(val) for val in overall_succ_10d_10t_10ht]
        overall_succ_15d_15t_15ht = [float(val) for val in overall_succ_15d_15t_15ht]
        overall_succ_20d_20t_20ht = [float(val) for val in overall_succ_20d_20t_20ht]
        
        avg_succ_5d_5t_5ht = sum(overall_succ_5d_5t_5ht) / float(len(overall_succ_5d_5t_5ht))
        avg_succ_10d_10t_10ht = sum(overall_succ_10d_10t_10ht) / float(len(overall_succ_10d_10t_10ht))
        avg_succ_15d_15t_15ht = sum(overall_succ_15d_15t_15ht) / float(len(overall_succ_15d_15t_15ht))
        avg_succ_20d_20t_20ht  = sum(overall_succ_20d_20t_20ht) / float(len(overall_succ_20d_20t_20ht))
        
        
        
        eval_sv_dict = {
            'avg_rot_diff': avg_rot_diff,
            'avg_trans_diff': avg_trans_diff,
            'final_rot_diff': final_rot_diff,
            'final_trans_diff': final_trans_diff,
            'avg_succ': avg_succ,
            'final_succ': final_succ,
            'hand_cd_err': hand_cd_err,
            # 'left_hand_cd_err': left_hand_cd_err,
            'transformed_gt_obj_verts': transformed_gt_obj_verts,
            'transformed_optimized_obj_verts': transformed_optimized_obj_verts,
            'tot_visual_pts': tot_visual_pts,
            'tot_gt_hand_pts': tot_gt_hand_pts,
            # 'left_tot_visual_pts': left_tot_visual_pts,
            # 'left_tot_gt_hand_pts': left_tot_gt_hand_pts
        }
        # np.save(os.path.join(save_eval_res_fn, "eval_sv_dict.npy"), eval_sv_dict)
        
        print(f"[Object] best_avg_angle_diff_quat: {best_avg_angle_diff_quat}, best_avg_diff_trans: {best_avg_diff_trans}, best_avg_succ_d5_t5: {best_avg_succ_d5_t5}, best_avg_succ_d10_t10: {best_avg_succ_d10_t10}, best_avg_succ_d15_t15: {best_avg_succ_d15_t15}, best_avg_succ_d20_t20: {best_avg_succ_d20_t20}")
        print(f"[Hand] best_avg_diff_robot_mano_fingers: {best_avg_diff_robot_mano_fingers}, best_avg_sim_vector_wrist_to_fingers: {best_avg_sim_vector_wrist_to_fingers}")
        print(f"[Hand & Object] avg_succ_5d_5t_5ht: {avg_succ_5d_5t_5ht}, avg_succ_10d_10t_10ht: {avg_succ_10d_10t_10ht}, avg_succ_15d_15t_15ht: {avg_succ_15d_15t_15ht}, avg_succ_20d_20t_20ht: {avg_succ_20d_20t_20ht}")
        
        succ_avg_5d_5t_5ht = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05 and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t_10ht = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t_15ht = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t_20ht = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 and best_avg_diff_robot_mano_fingers  <= 0.2
        
        succ_avg_5d_5t = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05  # and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 # and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 # and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 #  and best_avg_diff_robot_mano_fingers  <= 0.2
        
        print(f"[Hand Avg.] succ_avg_5d_5t: {succ_avg_5d_5t}, succ_avg_10d_10t: {succ_avg_10d_10t}, succ_avg_15d_15t: {succ_avg_15d_15t}, succ_avg_20d_20t: {succ_avg_20d_20t}")
        
        print(f"[Hand & Object Avg.] succ_avg_5d_5t_5ht: {succ_avg_5d_5t_5ht}, succ_avg_10d_10t_10ht: {succ_avg_10d_10t_10ht}, succ_avg_15d_15t_15ht: {succ_avg_15d_15t_15ht}, succ_avg_20d_20t_20ht: {succ_avg_20d_20t_20ht}")
        
        print(f"hand_cd_err: {hand_cd_err}")
    
    elif DATASET == "taco":
        dict_key = 'target'
        
        ##### 20231105_067 #####
        obj_name = "spoon2"
        obj_idxx = "20231105_067"
        obj_idxx_root = "20231105"
        TACO_DATA_ROOT = "/data2/datasets/xueyi/taco/processed_data"
        # ctl_vae_root = "/root/diffsim/control-vae-2"
        ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
        optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231105_067__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231105_067__mpc_v3_whandloss__step_0.npy"
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl" ## ref fn ->  with states information ##
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        
        gt_data_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231105_067_optmanorules_softv2_thres0d3_wveldp_/meshes/hand_obj_verts_faces_sv_dict_00569329.npy"
        ##### 20231105_067 #####
        
        
        ##### 20231105_067 #####
        obj_name = "brush2"
        obj_idxx = "20231104_017"
        obj_idxx_root = "20231104"
        TACO_DATA_ROOT = "/data3/datasets/xueyi/taco/processed_data"
        # ctl_vae_root = "/root/diffsim/control-vae-2"
        ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
        optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231104_017__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231104_017__mpc_v3_whandloss__step_0.npy"
        
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl" ## ref fn ->  with states information ##
        
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        
        gt_data_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231104_017_optmanorules_softv4_thres0d3_optveldp_/meshes/hand_obj_verts_faces_sv_dict_00905205.npy"
        ##### 20231105_067 #####
        
        
        ##### 20231104_035 #####
        obj_name = "sponge"
        obj_idxx = "20231104_035"
        obj_idxx_root = "20231104"
        TACO_DATA_ROOT = "/data2/datasets/xueyi/taco/processed_data"
        # ctl_vae_root = "/root/diffsim/control-vae-2"
        ctl_vae_root = "/home/xueyi/diffsim/Control-VAE"
        
        optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231104_035__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231104_035__mpc_v3_whandloss__step_0.npy"
        
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl" ## ref fn ->  with states information ##
        
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        
        gt_data_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231104_035_optmanorules_softv0_thres0d0_optrobo_/meshes/hand_obj_verts_faces_sv_dict_01052079.npy"
        ##### 20231104_035 #####
        
        ##### 20230919_043 ##### # 
        obj_name = "brush"
        obj_idxx = "20230919_043"
        obj_idxx_root = "20230919"
        
        gt_data_fn = "/data3/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230919_043_optmanorules_softv3_thres0d3_/meshes/hand_obj_verts_faces_sv_dict_01928209.npy"
        
        optimized_fn = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20230919_043__mpc_v3_whandloss_/evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20230919_043__mpc_v3_whandloss__step_0.npy"
        
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl" ## ref fn ->  with states information ##
        
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        ##### 20230919_043 #####
        
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_167_hammer167_wact_std0d01_netv1_mass10000_new_dp1d0_dtv2tsv2ctlv2_netv3optt_lstd_langdamp_optrules_mass2e5_trycma_stctl_adp0d7_mass5000__step"
        
        ### optimized folder and optimized fn prefix ###
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20230919_043__mpc_v3_whandloss_"
        optimized_fn_prefix = "evalulated_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20230919_043__mpc_v3_whandloss__step"
        
        optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20230919_043__mpc_v3_whandloss__step"
        
        
        ####### 20231104_017 #########
        # obj_name = "brush"
        obj_idxx = "20231104_017"
        obj_idxx_root = "20231104"
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231104_017__mpc_v3_whandloss_"
        
        optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231104_017__mpc_v3_whandloss__step"
        ####### 20231104_017 #########
        
        ####### 20231104_035 #########
        obj_idxx = "20231104_035"
        obj_idxx_root = "20231104"
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231104_035__mpc_v3_whandloss_"
        optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231104_035__mpc_v3_whandloss__step"
        ####### 20231104_035 #########
        
        
        ####### 20231105_067 #########
        obj_idxx = "20231105_067"
        obj_idxx_root = "20231105"
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        
        optimized_folder = "/data3/datasets/xueyi/control-vae/exp/subiters1024_optim_params_shadow_20231105_067__mpc_v3_whandloss_optobj_"
        optimized_fn_prefix = "evalulated_mpc_traj_sm_l512_wana_v3_subiters1024_optim_params_shadow_20231105_067__mpc_v3_whandloss_optobj__step"
        ####### 20231105_067 #########
        
        
        # 20231027 - 002
        obj_idxx = "20230923_001"
        obj_idxx_root = "20230923"
        obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230923_001_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230923_001_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        # 20231027 - 007 ##
        # obj_idxx = "20230923_007"
        # obj_idxx_root = "20230923"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230923_007_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230923_007_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        # 20231027 - 002
        # obj_idxx = "20230926_002"
        # obj_idxx_root = "20230926"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_002_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000043.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_002_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        
        # 20231027 - 002
        # obj_idxx = "20230926_035"
        # obj_idxx_root = "20230926"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_035_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230926_035_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        # obj_idxx = "20230927_023"
        # obj_idxx_root = "20230927"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_023_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_023_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        
        # obj_idxx = "20230927_031"
        # obj_idxx_root = "20230927"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data3/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_031_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230927_031_thres0d3_multi_stages_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        # obj_idxx = "20230928_025"
        # obj_idxx_root = "20230928"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_025_thres0d3_/thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/meshes/00000000.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_025_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        
        
        # obj_idxx = "20230928_027"
        # obj_idxx_root = "20230928"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_027_thres0d3_/thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230928_027_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        
        
        # obj_idxx = "20230929_002"
        # obj_idxx_root = "20230929"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_002_thres0d3_/thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_002_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        
        
        # obj_idxx = "20230929_005"
        # obj_idxx_root = "20230929"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_005_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_005_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        
        # obj_idxx = "20230929_018"
        # obj_idxx_root = "20230929"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_018_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000149.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20230929_018_thres0d3_/thres0d3_robo_thres0d25_robo_thres0d2_robo_thres0d1_robo_/meshes/00000000.npy"
        
        
        # obj_idxx = "20231031_157"
        # obj_idxx_root = "20231031"
        # obj_fn = f"{TACO_DATA_ROOT}/{obj_idxx_root}/right_{obj_idxx}.obj"
        # gt_ref_fn = f"/data2/datasets/xueyi/taco/processed_data/{obj_idxx_root}/right_{obj_idxx}.pkl"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231031_157_thres0d3_/thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_robo_/meshes/00000087.npy"
        # optimized_fn = "/data2/datasets/xueyi/neus/exp/hand_test_routine_2_light_color_wtime_active_passive/wmask_reverse_value_totviews_tag_train_retargeted_shadow_hand_states_optrobot__seq_taco_20231031_157_thres0d3_/thres0d25_robo_thres0d2_robo_thres0d1_robo_thres0d0_/meshes/00085347.npy"
        
        ## starting ##
        
        #### starting and ending object indexes ####
        opt_st_idx = 0
        opt_ed_idx = 60
        #### starting and ending object indexes ####
        
        
        #### best evaluated metrics ####
        minn_avg_rot_diff = 1000
        minn_avg_trans_diff = 1000
        
        minn_final_rot_diff = 1000
        minn_final_trans_diff = 1000
        
        maxx_avg_succ = 0.0
        maxx_final_succ = 0.0
        
        minn_avg_diff_obj_err = 1000
        
        best_transformed_gt_obj_verts = None
        best_transformed_optimized_obj_verts = None
        
        minn_avg_obj_state_diff = 1000
        
        
        minn_avg_diff_obj_err = 1000
        
        best_transformed_gt_obj_verts = None
        best_transformed_optimized_obj_verts = None
        
        minn_avg_obj_state_diff = 1000
        
        # avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20
        best_avg_angle_diff_quat = 1000.0
        best_avg_diff_trans = 1000.0
        best_avg_succ_d5_t5 = 0.0 # 
        best_avg_succ_d10_t10 = 0.0
        best_avg_succ_d15_t15 = 0.0
        best_avg_succ_d20_t20 = 0.0
        best_avg_diff_robot_mano_fingers = 1000.0
        best_avg_sim_vector_wrist_to_fingers = 0.0
        
        
        ### best evalulated index ###
        best_eval_idx = None
        
        
        dict_key = 'target'
        
        
        
        
        avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts, avg_diff_obj_err, avg_angle_diff_quat, avg_diff_trans, avg_succ_d5_t5, avg_succ_d10_t10, avg_succ_d15_t15, avg_succ_d20_t20,  tot_angle_diff_quat, tot_diff_trans  = calc_obj_metrics_taco_v2(optimized_fn, gt_ref_fn, dict_key='state', obj_fn=obj_fn)
        
        
        if avg_diff_obj_err < minn_avg_diff_obj_err:
            minn_avg_diff_obj_err = avg_diff_obj_err
            minn_avg_rot_diff = avg_rot_diff
            minn_avg_trans_diff = avg_trans_diff
            
            minn_final_rot_diff = final_rot_diff
            minn_final_trans_diff = final_trans_diff
            
            
            maxx_avg_succ = avg_succ
            maxx_final_succ = final_succ
            
            best_transformed_gt_obj_verts = transformed_gt_obj_verts
            best_transformed_optimized_obj_verts = transformed_optimized_obj_verts
            
            best_avg_angle_diff_quat = avg_angle_diff_quat
            best_avg_diff_trans = avg_diff_trans
            best_avg_succ_d5_t5 = avg_succ_d5_t5
            best_avg_succ_d10_t10 = avg_succ_d10_t10
            best_avg_succ_d15_t15 = avg_succ_d15_t15
            best_avg_succ_d20_t20 = avg_succ_d20_t20
            
            
            hand_cd_err, tot_visual_pts, tot_gt_hand_pts, avg_diff_robot_mano_fingers, avg_sim_vector_wrist_to_fingers, tot_diff_robot_mano_fingers, tot_sim_vector_wrist_to_fingers = calcu_hand_tracking_errors_taco_v2(optimized_fn, gt_ref_fn,  dict_key=dict_key)
            
            best_eval_idx = 0
        
            overall_succ_5d_5t_5ht = [tot_angle_diff_quat[i_fr] <= 5.0 and tot_diff_trans[i_fr] <= 0.05 and tot_diff_robot_mano_fingers[i_fr] <= 0.05 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_10d_10t_10ht = [tot_angle_diff_quat[i_fr] <= 10.0 and tot_diff_trans[i_fr] <= 0.1 and tot_diff_robot_mano_fingers[i_fr] <= 0.1 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_15d_15t_15ht = [tot_angle_diff_quat[i_fr] <= 15.0 and tot_diff_trans[i_fr] <= 0.15 and tot_diff_robot_mano_fingers[i_fr] <= 0.15 for i_fr in range(len(tot_angle_diff_quat))]
            overall_succ_20d_20t_20ht = [tot_angle_diff_quat[i_fr] <= 20.0 and tot_diff_trans[i_fr] <= 0.2 and tot_diff_robot_mano_fingers[i_fr] <= 0.2 for i_fr in range(len(tot_angle_diff_quat))]
            
            best_avg_diff_robot_mano_fingers = avg_diff_robot_mano_fingers
            best_avg_sim_vector_wrist_to_fingers = avg_sim_vector_wrist_to_fingers
        
        
        
        # for opt_idx in range(opt_st_idx, opt_ed_idx + 1):
            
        #     #### get the optimized file path ####
        #     cur_opt_fn = os.path.join(optimized_folder, f"{optimized_fn_prefix}_{opt_idx}.npy") 
        
        #     print(f"cur_opt_fn: {cur_opt_fn}")
        
        #     #### get the object metric ####
        #     avg_rot_diff, avg_trans_diff, final_rot_diff, final_trans_diff, avg_succ, final_succ, transformed_gt_obj_verts, transformed_optimized_obj_verts , avg_diff_obj_err  = calc_obj_metrics_taco(cur_opt_fn, gt_ref_fn, dict_key=dict_key, obj_fn=obj_fn)
            
        #     # #### get the hand metric ####
        #     # hand_cd_err, tot_visual_pts, tot_gt_hand_pts = calcu_hand_tracking_errors_taco(cur_opt_fn, gt_data_fn,  dict_key=dict_key)
            
            
        #     if avg_diff_obj_err < minn_avg_diff_obj_err:
        #         minn_avg_diff_obj_err = avg_diff_obj_err
        #         minn_avg_rot_diff = avg_rot_diff
        #         minn_avg_trans_diff = avg_trans_diff
                
        #         minn_final_rot_diff = final_rot_diff
        #         minn_final_trans_diff = final_trans_diff
                
                
        #         maxx_avg_succ = avg_succ
        #         maxx_final_succ = final_succ
                
        #         best_transformed_gt_obj_verts = transformed_gt_obj_verts
        #         best_transformed_optimized_obj_verts = transformed_optimized_obj_verts
            
        #         hand_cd_err, tot_visual_pts, tot_gt_hand_pts = calcu_hand_tracking_errors_taco(cur_opt_fn, gt_ref_fn,  dict_key=dict_key)
                
        #         best_eval_idx = opt_idx
        
        
        overall_succ_5d_5t_5ht = [float(val) for val in overall_succ_5d_5t_5ht]
        overall_succ_10d_10t_10ht = [float(val) for val in overall_succ_10d_10t_10ht]
        overall_succ_15d_15t_15ht = [float(val) for val in overall_succ_15d_15t_15ht]
        overall_succ_20d_20t_20ht = [float(val) for val in overall_succ_20d_20t_20ht]
        
        avg_succ_5d_5t_5ht = sum(overall_succ_5d_5t_5ht) / float(len(overall_succ_5d_5t_5ht))
        avg_succ_10d_10t_10ht = sum(overall_succ_10d_10t_10ht) / float(len(overall_succ_10d_10t_10ht))
        avg_succ_15d_15t_15ht = sum(overall_succ_15d_15t_15ht) / float(len(overall_succ_15d_15t_15ht))
        avg_succ_20d_20t_20ht  = sum(overall_succ_20d_20t_20ht) / float(len(overall_succ_20d_20t_20ht))
        
        
        
        # left_hand_cd_err, left_tot_visual_pts, left_tot_gt_hand_pts = calcu_hand_tracking_errors_lefthand(optimized_fn, gt_data_fn,  dict_key=dict_key)
        eval_sv_dict = {
            'avg_rot_diff': avg_rot_diff,
            'avg_trans_diff': avg_trans_diff,
            'final_rot_diff': final_rot_diff,
            'final_trans_diff': final_trans_diff,
            'avg_succ': avg_succ,
            'final_succ': final_succ,
            'hand_cd_err': hand_cd_err,
            'transformed_gt_obj_verts': transformed_gt_obj_verts,
            'transformed_optimized_obj_verts': transformed_optimized_obj_verts,
            'tot_visual_pts': tot_visual_pts,
            'tot_gt_hand_pts': tot_gt_hand_pts,
        }
        
        
        save_eval_res_fn = os.path.join(ctl_vae_root, "eval_res", "taco", obj_name)
        os.makedirs(save_eval_res_fn, exist_ok=True)
        
        
        sv_dict_fn = f"eval_sv_dict_taco_{obj_idxx}.npy"
        sv_dict_fn = os.path.join(save_eval_res_fn, sv_dict_fn)
        np.save(sv_dict_fn, eval_sv_dict)
        
        
        ##### hand cd loss and the sv_dict_fn #####
        print(f"Best eval idx: {best_eval_idx}, minn_avg_diff_obj_err: {minn_avg_diff_obj_err}, avg_rot_diff: {minn_avg_rot_diff}, avg_trans_diff: {minn_avg_trans_diff}, final_rot_diff: {minn_final_rot_diff}, final_trans_diff: {minn_final_trans_diff}, avg_succ: {maxx_avg_succ}, final_succ: {maxx_final_succ}")
        
        
        
        print(f"[Object] best_avg_angle_diff_quat: {best_avg_angle_diff_quat}, best_avg_diff_trans: {best_avg_diff_trans}, best_avg_succ_d5_t5: {best_avg_succ_d5_t5}, best_avg_succ_d10_t10: {best_avg_succ_d10_t10}, best_avg_succ_d15_t15: {best_avg_succ_d15_t15}, best_avg_succ_d20_t20: {best_avg_succ_d20_t20}")
        print(f"[Hand] best_avg_diff_robot_mano_fingers: {best_avg_diff_robot_mano_fingers}, best_avg_sim_vector_wrist_to_fingers: {best_avg_sim_vector_wrist_to_fingers}")
        print(f"[Hand & Object] avg_succ_5d_5t_5ht: {avg_succ_5d_5t_5ht}, avg_succ_10d_10t_10ht: {avg_succ_10d_10t_10ht}, avg_succ_15d_15t_15ht: {avg_succ_15d_15t_15ht}, avg_succ_20d_20t_20ht: {avg_succ_20d_20t_20ht}")
        
        succ_avg_5d_5t_5ht = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05 and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t_10ht = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t_15ht = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t_20ht = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 and best_avg_diff_robot_mano_fingers  <= 0.2
        
        succ_avg_5d_5t = best_avg_angle_diff_quat <= 5.0 and best_avg_diff_trans <= 0.05  # and best_avg_diff_robot_mano_fingers <= 0.05
        succ_avg_10d_10t = best_avg_angle_diff_quat <= 10.0 and  best_avg_diff_trans <= 0.1 # and best_avg_diff_robot_mano_fingers <= 0.1
        succ_avg_15d_15t = best_avg_angle_diff_quat <= 15.0 and best_avg_diff_trans <= 0.15 # and best_avg_diff_robot_mano_fingers  <= 0.15
        succ_avg_20d_20t = best_avg_angle_diff_quat <= 20.0 and best_avg_diff_trans <= 0.2 #  and best_avg_diff_robot_mano_fingers  <= 0.2
        
        print(f"[Hand Avg.] succ_avg_5d_5t: {succ_avg_5d_5t}, succ_avg_10d_10t: {succ_avg_10d_10t}, succ_avg_15d_15t: {succ_avg_15d_15t}, succ_avg_20d_20t: {succ_avg_20d_20t}")
        
        print(f"[Hand & Object Avg.] succ_avg_5d_5t_5ht: {succ_avg_5d_5t_5ht}, succ_avg_10d_10t_10ht: {succ_avg_10d_10t_10ht}, succ_avg_15d_15t_15ht: {succ_avg_15d_15t_15ht}, succ_avg_20d_20t_20ht: {succ_avg_20d_20t_20ht}")
        
        
        
        print(f"hand_cd_err: {hand_cd_err}")
        
        # print(f"avg_rot_diff: {avg_rot_diff}, avg_trans_diff: {avg_trans_diff}, final_rot_diff: {final_rot_diff}, final_trans_diff: {final_trans_diff}, avg_succ: {avg_succ}, final_succ: {final_succ}")
        # print(f"hand_cd_err: {hand_cd_err}")
    
    exit(0)

