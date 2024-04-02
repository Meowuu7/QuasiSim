import os
# from utils.scandir import scandir
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import torch
import json
from time import time, sleep
from manopth.manolayer import ManoLayer
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import (PerspectiveCameras, PointLights, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, SoftPhongShader, TexturesVertex)
import open3d as o3d
import shutil
from collections import OrderedDict

def load_simplied_nokov_objs_mesh(root, video_id, frame_list, use_cm = True):
    '''

    return: tool_verts_batch: [num_frame, num_verts, 3]
            tool_faces
            obj_verts_batch: [num_frame, num_verts, 3]
            obj_faces
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir)
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    frame_idx_list = [(int(frame)-1) for frame in frame_list]

    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2
    
    if use_cm:
        tool_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{tool_id}_cm.obj')
    else:
        tool_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{tool_id}_m.obj')
    tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
    tool_verts = np.asarray(tool_mesh.vertices)
    if use_cm:
        tool_verts = torch.from_numpy(tool_verts / 100.).float()
    else:
        tool_verts = torch.from_numpy(tool_verts).float()
    tool_faces = torch.from_numpy(np.asarray(tool_mesh.triangles))

    tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
    assert os.path.exists(tool_pose_path)
    tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))[frame_idx_list, ...]
    tool_verts_batch = verts_apply_pose_batch(tool_verts, tool_pose_batch)

    if use_cm:
        obj_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{obj_id}_cm.obj')
    else:
        obj_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{obj_id}_m.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_verts = np.asarray(obj_mesh.vertices)
    if use_cm:
        obj_verts = torch.from_numpy(obj_verts / 100.).float()
    else:
        obj_verts = torch.from_numpy(obj_verts).float()
    obj_faces = torch.from_numpy(np.asarray(obj_mesh.triangles))

    obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')
    assert os.path.exists(obj_pose_path)
    obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]
    obj_verts_batch = verts_apply_pose_batch(obj_verts, obj_pose_batch)

    return tool_verts_batch, tool_faces, obj_verts_batch, obj_faces

def load_simplied_objs_mesh(root, video_id, use_cm = True):
    '''
    templete

    return: tool_verts: [num_frame, num_verts, 3]
            tool_faces
            obj_verts: [num_frame, num_verts, 3]
            obj_faces
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir)
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2
    
    if use_cm:
        tool_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{tool_id}_cm.obj')
    else:
        tool_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{tool_id}_m.obj')
    tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
    tool_verts = np.asarray(tool_mesh.vertices)
    if use_cm:
        tool_verts = torch.from_numpy(tool_verts / 100.).float()
    else:
        tool_verts = torch.from_numpy(tool_verts).float()
    tool_faces = torch.from_numpy(np.asarray(tool_mesh.triangles))

    if use_cm:
        obj_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{obj_id}_cm.obj')
    else:
        obj_mesh_path = os.path.join(root, 'object_models_final_simplied', f'{obj_id}_m.obj')
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_verts = np.asarray(obj_mesh.vertices)
    if use_cm:
        obj_verts = torch.from_numpy(obj_verts / 100.).float()
    else:
        obj_verts = torch.from_numpy(obj_verts).float()
    obj_faces = torch.from_numpy(np.asarray(obj_mesh.triangles))

    return tool_verts, tool_faces, obj_verts, obj_faces

def load_objs_params(root, video_id, frame_list):
    '''

    return: tool_verts_batch: [num_frame, num_verts, 3]
            tool_faces
            obj_verts_batch: [num_frame, num_verts, 3]
            obj_faces
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir)
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    frame_idx_list = [(int(frame)-1) for frame in frame_list]

    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2
    
    tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
    assert os.path.exists(tool_pose_path)
    tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))[frame_idx_list, ...]
    tool_rot_batch = tool_pose_batch[:, :3, :3] # [num_frame, 3, 3]
    tool_trans_batch = tool_pose_batch[:, :3, 3] # [num_frame, 1, 3]

    obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')
    assert os.path.exists(obj_pose_path)
    obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]
    obj_rot_batch = obj_pose_batch[:, :3, :3] # [num_frame, 3, 3]
    obj_trans_batch = obj_pose_batch[:, :3, 3] # [num_frame, 1, 3]
    
    return tool_rot_batch, tool_trans_batch, obj_rot_batch, obj_trans_batch


def load_bg_img(root, video_id, camera_id, frame_id):
    img_root = os.path.join(root, video_id, 'imgs')
    # os.path.exists(img_root)
    path = os.path.join(img_root, camera_id, camera_id + '_' + frame_id + '.png')
    # assert os.path.exists(path), path
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len = 2):
    
    represent_relation = OrderedDict()
    # represent_order = [] # 防止低版本python的字典是无序的，但似乎用OrderedDict就好了
    for frame in frame_list:
        num = int(frame)
        if num <= first_batch_len:
            key = str(1).zfill(5)
        else:
            key = str(((num - (first_batch_len + 1)) // BATCH_SIZE) * BATCH_SIZE + first_batch_len + 1).zfill(5)
        if key not in represent_relation:
            represent_relation[key] = []
        represent_relation[key].append(frame)
            
    return represent_relation

def load_bg_img_acc(root, video_id, camera_id, BATCH_SIZE, frame_list):
    date = video_id[:8]
    sub_video_root = os.path.join(root, date, video_id, 'sub_video')
    
    represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list)

    full_img_list = []
    for represent_frame_id, represented_frame_list in represent_relation.items():
        
        video_path = os.path.join(sub_video_root, camera_id, camera_id + '_' + represent_frame_id + '.mp4')

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        suc = cap.isOpened()
        img_list = []
        for i in range(BATCH_SIZE):
            suc, img = cap.read()
            if not suc:
                break
            img_list.append(img)
        
        # num_img = len(img_list)

        img_list_ = []
        for idx in [int(frame) - int(represent_frame_id) for frame in represented_frame_list]:
            # assert idx >= 0 and idx < num_img
            img_list_.append(img_list[idx])
        img_list = img_list_

        for img in img_list:
            full_img_list.append(img)

    return full_img_list

def load_bg_imgs(root, video_id, frame_list, camera_list, BATCH_SIZE):
    
    rgb_batch = [[] for _ in range(len(frame_list))]
    
    for c_idx, camera in enumerate(camera_list):
        img_list = load_bg_img_acc(root, video_id, camera, BATCH_SIZE, frame_list)
        for f_idx, bg in enumerate(img_list):
            rgb_batch[f_idx].append(bg)
    
    return rgb_batch

def load_crop_info_test(camera_id, frame_id, is_righthand):
    '''

    return:

    '''
    if is_righthand:
        crop_info_path = os.path.join('/home/hlyang/utils/HOI/img/', camera_id, 'crop_imgs_righthand', camera_id + '_' + frame_id + '_crop_info.pkl')
    else:
        crop_info_path = os.path.join('/home/hlyang/utils/HOI/img/', camera_id, 'crop_imgs_lefthand', camera_id + '_' + frame_id + '_crop_info.pkl')
    assert os.path.exists(crop_info_path)

    with open(crop_info_path, 'rb') as f:
        crop_info = pickle.load(f)
    h_min, h_max, w_min, w_max, h_mean, w_mean = crop_info
    return h_min, h_max, w_min, w_max, h_mean, w_mean

def load_crop_info(root, video_id, camera_id, frame_id, is_righthand):
    '''
    return:

    '''
    if is_righthand:
        crop_info_path = os.path.join(root, video_id, 'crop', 'right_hand', camera_id, camera_id + '_' + frame_id + '_crop_info.pkl')
    else:
        crop_info_path = os.path.join(root, video_id, 'crop', 'left_hand', camera_id, camera_id + '_' + frame_id + '_crop_info.pkl')
    assert os.path.exists(crop_info_path)

    with open(crop_info_path, 'rb') as f:
        crop_info = pickle.load(f)
    h_min, h_max, w_min, w_max, h_mean, w_mean = crop_info
    return h_min, h_max, w_min, w_max, h_mean, w_mean

def load_crop_info_v2(root, date, video_id, from_exp_name, camera_list, frame_list, right_hand_bool, BATCH_SIZE = 20, represent_frame_id = None):
    '''
    TODO: 考虑缺失的情况，但应该不会有缺失的情况。现在的处理方案：只载入读到了，空的跳过

    '''
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}
    
    crop_info_batch = {}
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        if right_hand_bool:
            path = os.path.join(root, date, video_id, from_exp_name, 'right_hand', represent_frame_id + '_crop_info.pkl')
        else:
            path = os.path.join(root, date, video_id, from_exp_name, 'left_hand', represent_frame_id + '_crop_info.pkl')
        with open(path, 'rb') as f:
            crop_info_ = pickle.load(f)
            
            crop_info_batch.update(crop_info_)
            
        # for f_idx, frame_id in enumerate(represented_frame_list):
        #     crop_info_batch[frame_id] = {}
            
        #     crop_info_frame_ = crop_info_.get(frame_id, None)
        #     if crop_info_frame_ is None:
        #         continue
            
        #     for c_idx, camera_id in enumerate(camera_list):
        #         crop_info_camera_ = crop_info_frame_.get(camera_id, None)
        #         if crop_info_camera_ is not None:
        #             crop_info_batch[frame_id][camera_id] = crop_info_camera_
                    
    return crop_info_batch
                    
                    
                    

    h_min, h_max, w_min, w_max, h_mean, w_mean = crop_info
    return h_min, h_max, w_min, w_max, h_mean, w_mean

def get_seg_infos_batch_test(seg_path_prefix, frame_list, camera_list):
    '''
    return:
    object_seg, left_hand_seg, right_hand_seg 形如[batch_size, num_camera, h, w]
    '''
    object_seg_batch = []
    left_hand_seg_batch = []
    right_hand_seg_batch = []
    for frame_id in frame_list:
        object_seg_cameras = []
        left_hand_seg_cameras = []
        right_hand_seg_cameras = []
        for camera_id in camera_list:
            # path = os.path.join(seg_path_prefix, camera_id, 'denoised_mask_imgs', camera_id + '_' + frame_id + '_mask_denoised.png')
            path = os.path.join(seg_path_prefix, camera_id, 'mask_imgs', camera_id + '_' + frame_id + '_mask.png')
            assert os.path.exists(path)
            seg_img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
            seg_img = torch.from_numpy(seg_img)
            # object_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] > 50) & (seg_img[:, :, 2] > 50), True, False)
            # left_hand_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] > 50) & (seg_img[:, :, 2] < 50), True, False)
            # right_hand_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] < 50) & (seg_img[:, :, 2] > 50), True, False)
            object_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] > 50) & (seg_img[:, :, 2] > 50), 1.0, 0.0)
            left_hand_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] > 50) & (seg_img[:, :, 2] < 50), 1.0, 0.0)
            right_hand_seg = torch.where((seg_img[:, :, 0] < 50) & (seg_img[:, :, 1] < 50) & (seg_img[:, :, 2] > 50), 1.0, 0.0)
            object_seg_cameras.append(object_seg)
            left_hand_seg_cameras.append(left_hand_seg)
            right_hand_seg_cameras.append(right_hand_seg)
        object_seg_cameras = torch.stack(object_seg_cameras)
        left_hand_seg_cameras = torch.stack(left_hand_seg_cameras)
        right_hand_seg_cameras = torch.stack(right_hand_seg_cameras)
        object_seg_batch.append(object_seg_cameras)
        left_hand_seg_batch.append(left_hand_seg_cameras)
        right_hand_seg_batch.append(right_hand_seg_cameras)
    object_seg_batch = torch.stack(object_seg_batch)
    left_hand_seg_batch = torch.stack(left_hand_seg_batch)
    right_hand_seg_batch = torch.stack(right_hand_seg_batch)
    return object_seg_batch, left_hand_seg_batch, right_hand_seg_batch

def get_seg_infos_batch(root, video_id, frame_list, camera_list):
    '''
    return:
    left_hand_seg, right_hand_seg, object1_seg, object2_seg 形如[batch_size, num_camera, h, w]
    '''
    left_hand_seg_batch = []
    right_hand_seg_batch = []
    object1_seg_batch = []
    object2_seg_batch = []

    mask_dict = {}
    for camera_id in camera_list:
        path = os.path.join(root, video_id, 'anno_results', video_id+'|'+camera_id+'.npy')
        mask = np.load(path)
        mask_dict[camera_id] = torch.from_numpy(mask)

    #如果frame_list是连续的，可以直接索引少一层循环，速度应该会快得多。
    frame_idx_list = [int(frame_id)-1 for frame_id in frame_list]

    continuous_bool = True
    for i in range(len(frame_idx_list)-1):
        if frame_idx_list[i] + 1 != frame_idx_list[i+1]:
            continuous_bool = False
            break
    with torch.no_grad():
        if continuous_bool:
            frame_start = frame_idx_list[0]
            frame_end = frame_idx_list[-1]

            for camera_id in camera_list:
                mask = mask_dict[camera_id]

                left_hand_seg = torch.where(mask[frame_start: frame_end + 1, ...] == 1, 1., 0.)
                right_hand_seg = torch.where(mask[frame_start: frame_end + 1, ...] == 2, 1., 0.)
                object1_seg = torch.where(mask[frame_start: frame_end + 1, ...] == 3, 1., 0.)
                object2_seg = torch.where(mask[frame_start: frame_end + 1, ...] == 4, 1., 0.)

                left_hand_seg_batch.append(left_hand_seg)
                right_hand_seg_batch.append(right_hand_seg)
                object1_seg_batch.append(object1_seg)
                object2_seg_batch.append(object2_seg)

            left_hand_seg_batch = torch.stack(left_hand_seg_batch).transpose(0, 1)
            right_hand_seg_batch = torch.stack(right_hand_seg_batch).transpose(0, 1)
            object1_seg_batch = torch.stack(object1_seg_batch).transpose(0, 1)
            object2_seg_batch = torch.stack(object2_seg_batch).transpose(0, 1)
        else:
            for frame_id in frame_list:
                frame_idx = int(frame_id) - 1
                left_hand_seg_cameras = []
                right_hand_seg_cameras = []
                object1_seg_cameras = []
                object2_seg_cameras = []
                for camera_id in camera_list:
                    mask = mask_dict[camera_id]
                    left_hand_seg = torch.where(mask[frame_idx, ...] == 1, 1., 0.)
                    right_hand_seg = torch.where(mask[frame_idx, ...] == 2, 1., 0.)
                    object1_seg = torch.where(mask[frame_idx, ...] == 3, 1., 0.)
                    object2_seg = torch.where(mask[frame_idx, ...] == 4, 1., 0.)

                    left_hand_seg_cameras.append(left_hand_seg)
                    right_hand_seg_cameras.append(right_hand_seg)
                    object1_seg_cameras.append(object1_seg)
                    object2_seg_cameras.append(object2_seg)

                left_hand_seg_cameras = torch.stack(left_hand_seg_cameras)
                right_hand_seg_cameras = torch.stack(right_hand_seg_cameras)
                object1_seg_cameras = torch.stack(object1_seg_cameras)
                object2_seg_cameras = torch.stack(object2_seg_cameras)

                left_hand_seg_batch.append(left_hand_seg_cameras)
                right_hand_seg_batch.append(right_hand_seg_cameras)
                object1_seg_batch.append(object1_seg_cameras)
                object2_seg_batch.append(object2_seg_cameras)

            left_hand_seg_batch = torch.stack(left_hand_seg_batch)
            right_hand_seg_batch = torch.stack(right_hand_seg_batch)
            object1_seg_batch = torch.stack(object1_seg_batch)
            object2_seg_batch = torch.stack(object2_seg_batch)

    return left_hand_seg_batch, right_hand_seg_batch, object1_seg_batch, object2_seg_batch

def get_seg_infos_batch2(root, video_id, frame_list, camera_list):
    '''
    get_seg_infos_batch1的numpy实现
    TODO：不知为何性能非常差

    return:
    left_hand_seg, right_hand_seg, object1_seg, object2_seg 形如[batch_size, num_camera, h, w]
    '''
    left_hand_seg_batch = []
    right_hand_seg_batch = []
    object1_seg_batch = []
    object2_seg_batch = []

    mask_dict = {}
    for camera_id in camera_list:
        path = os.path.join(root, video_id, 'anno_results', video_id+'|'+camera_id+'.npy')
        mask = np.load(path)
        mask_dict[camera_id] = mask

    #如果frame_list是连续的，可以直接索引少一层循环，速度应该会快得多。
    frame_idx_list = [int(frame_id)-1 for frame_id in frame_list]

    continuous_bool = True
    for i in range(len(frame_idx_list)-1):
        if frame_idx_list[i] + 1 != frame_idx_list[i+1]:
            continuous_bool = False
            break

    if continuous_bool:
        frame_start = frame_idx_list[0]
        frame_end = frame_idx_list[-1]

        for camera_id in camera_list:
            print('hello_' + camera_id)
            mask = mask_dict[camera_id]

            left_hand_seg = np.where(mask[frame_start: frame_end + 1, ...] == 1, 1., 0.)
            right_hand_seg = np.where(mask[frame_start: frame_end + 1, ...] == 2, 1., 0.)
            object1_seg = np.where(mask[frame_start: frame_end + 1, ...] == 3, 1., 0.)
            object2_seg = np.where(mask[frame_start: frame_end + 1, ...] == 4, 1., 0.)

            left_hand_seg_batch.append(left_hand_seg)
            right_hand_seg_batch.append(right_hand_seg)
            object1_seg_batch.append(object1_seg)
            object2_seg_batch.append(object2_seg)

        left_hand_seg_batch = np.stack(left_hand_seg_batch).transpose(0, 1)
        right_hand_seg_batch = np.stack(right_hand_seg_batch).transpose(0, 1)
        object1_seg_batch = np.stack(object1_seg_batch).transpose(0, 1)
        object2_seg_batch = np.stack(object2_seg_batch).transpose(0, 1)
    else:
        for frame_id in frame_list:
            frame_idx = int(frame_id) - 1
            left_hand_seg_cameras = []
            right_hand_seg_cameras = []
            object1_seg_cameras = []
            object2_seg_cameras = []
            for camera_id in camera_list:
                mask = mask_dict[camera_id]
                left_hand_seg = np.where(mask[frame_idx, ...] == 1, 1., 0.)
                right_hand_seg = np.where(mask[frame_idx, ...] == 2, 1., 0.)
                object1_seg = np.where(mask[frame_idx, ...] == 3, 1., 0.)
                object2_seg = np.where(mask[frame_idx, ...] == 4, 1., 0.)

                left_hand_seg_cameras.append(left_hand_seg)
                right_hand_seg_cameras.append(right_hand_seg)
                object1_seg_cameras.append(object1_seg)
                object2_seg_cameras.append(object2_seg)

            left_hand_seg_cameras = np.stack(left_hand_seg_cameras)
            right_hand_seg_cameras = np.stack(right_hand_seg_cameras)
            object1_seg_cameras = np.stack(object1_seg_cameras)
            object2_seg_cameras = np.stack(object2_seg_cameras)

            left_hand_seg_batch.append(left_hand_seg_cameras)
            right_hand_seg_batch.append(right_hand_seg_cameras)
            object1_seg_batch.append(object1_seg_cameras)
            object2_seg_batch.append(object2_seg_cameras)

        left_hand_seg_batch = np.stack(left_hand_seg_batch)
        right_hand_seg_batch = np.stack(right_hand_seg_batch)
        object1_seg_batch = np.stack(object1_seg_batch)
        object2_seg_batch = np.stack(object2_seg_batch)

    return left_hand_seg_batch, right_hand_seg_batch, object1_seg_batch, object2_seg_batch

def get_seg_infos_batch3(root, video_id: str, frame_list, camera_list):
    '''

    return:
    seg: 形如[batch_size, num_camera, h, w]，其中值为1 2 3 4分别代表左手、右手、object1和object2
    '''
    seg_batch = []

    mask_dict = {}
    for camera_id in camera_list:
        path = os.path.join(root, video_id, 'anno_results', video_id+'|'+camera_id+'.npy')
        mask = np.load(path)
        mask_dict[camera_id] = mask

    #如果frame_list是连续的，可以直接索引少一层循环，速度应该会快得多。
    frame_idx_list = [int(frame_id)-1 for frame_id in frame_list]

    continuous_bool = True
    for i in range(len(frame_idx_list)-1):
        if frame_idx_list[i] + 1 != frame_idx_list[i+1]:
            continuous_bool = False
            break

    if continuous_bool:
        frame_start = frame_idx_list[0]
        frame_end = frame_idx_list[-1] + 1

        for camera_id in camera_list:
            mask = mask_dict[camera_id]

            seg = mask[frame_start: frame_end, ...]
            seg_batch.append(seg)

        seg_batch = np.stack(seg_batch)
        seg_batch = seg_batch.swapaxes(0, 1)
    else:
        for frame_idx in frame_idx_list:
            seg_cameras = []
            for camera_id in camera_list:
                mask = mask_dict[camera_id]
                seg = [frame_idx, ...]

                seg_cameras.append(seg)

            seg_cameras = np.stack(seg_cameras)

            seg_batch.append(seg_cameras)

        seg_batch = np.stack(seg_batch)

    return seg_batch

def get_downsampled_seg_infos_batch(root, video_id: str, frame_list, camera_list):
    '''

    return: (seg, downsample_factor)
    seg: 形如[batch_size, num_camera, h, w]，其中值为1 2 3 4分别代表右手、左手、object1(右手操作的物体)和object2(左手操作的物体), h为750, w为1024
    downsample_factor: 4
    '''
    seg_batch = []

    for frame_id in frame_list:
        seg_camera = []
        for camera_id in camera_list:
            path = os.path.join(root, video_id, 'anno_results', camera_id, camera_id + '_' + frame_id+'.npy')
            assert os.path.exists(path)
            seg = np.load(path)
            seg_camera.append(seg)
        seg_camera = np.stack(seg_camera)
        seg_batch.append(seg_camera)
    seg_batch = np.stack(seg_batch)

    return seg_batch, 4

def get_downsampled_seg_infos_batch_v2(root, video_id: str, from_exp_name, frame_list, camera_list):
    '''
    某些视角会track失败，得不到mask：如果path不存在，就生成一个全0。

    return: (seg, downsample_factor)
    seg: 形如[batch_size, num_camera, h, w]，其中值为1 2 3 4分别代表右手、左手、object1(右手操作的物体)和object2(左手操作的物体), h为750, w为1024
    downsample_factor: 4
    '''
    seg_batch = []

    for frame_id in frame_list:
        seg_camera = []
        for camera_id in camera_list:
            path = os.path.join(root, video_id, from_exp_name, 'res', camera_id, camera_id + '_' + frame_id+'.npy')
            if os.path.exists(path):
                seg = np.load(path)
            else:
                seg = np.zeros(shape=(750, 1024), dtype=np.uint8)
            seg_camera.append(seg)
        seg_camera = np.stack(seg_camera)
        seg_batch.append(seg_camera)
    seg_batch = np.stack(seg_batch)

    return seg_batch, 4

def get_downsampled_seg_infos_batch_v2_acc_batch(root, date, video_id: str, from_exp_name, frame_list, camera_list, BATCH_SIZE = 20, represent_frame_id = None):
    '''
    某些视角会track失败，得不到mask：如果path不存在，就生成一个全0。

    return: (full_mask, downsample_factor)
    full_mask: 形如[batch_size, num_camera, h, w]，其中值为1 2 3 4分别代表右手、左手、object1(右手操作的物体)和object2(左手操作的物体), h为750, w为1024
    downsample_factor: 4
    '''
    
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}
        
    seg_batch = []
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        path = os.path.join(root, date, video_id, from_exp_name, 'res', represent_frame_id+'.npy')
        mask = np.load(path)
        
        for f_idx, frame_id in enumerate(represented_frame_list):
            seg_camera = []
            for c_idx, camera_id in enumerate(camera_list):
                try:
                    seg = mask[f_idx, c_idx]
                except:
                    seg = np.zeros(shape=(750, 1024), dtype=np.uint8)
                seg_camera.append(seg)
            seg_camera = np.stack(seg_camera)
            seg_batch.append(seg_camera)
            
    seg_batch = np.stack(seg_batch)

    return seg_batch, 4

def get_downsampled_2d_mask(root, video_id: str, mask_type: str, frame_list, camera_list):
    '''
    用于fit_object_model中对物体进行监督
    20240129

    return: (full_mask, downsample_factor)
    full_mask: 形如[batch_size, num_camera, h, w]，其中值为1 2 3 4分别代表右手、左手、object1(右手操作的物体)和object2(左手操作的物体), h为750, w为1024
    downsample_factor: 4
    '''
    
    assert mask_type in ['masks', 'fake_masks']
    date = video_id[:8]
    
    dir = os.path.join('/share1/datasets/HOI-mocap/2Dmask_results_final', date, video_id)
    
    frame_idx_list = [(int(frame)-1) for frame in frame_list]
    seg_batch = []
    for camera in camera_list:
    
        path = os.path.join(dir, f'{camera}_{mask_type}.npy')
        mask = np.load(path)
        
        seg_batch.append(mask[frame_idx_list])
            
    seg_batch = np.stack(seg_batch)
    seg_batch = np.swapaxes(seg_batch, 0, 1)

    return seg_batch, 4

def cvt_multiview_imgs2mp4(img_dir:str, save_path, save_fps, camera_list, frame_list, height = 3000, width = 4096, downsample_factor = 4):
    # camera_img_dict = {}
    # paths = list(scandir(img_prefix, suffix='.png', recursive=True, full_path=True))
    # for path in paths:
    #     dirname, basename = os.path.split(path)
    #     camera_id = basename.split('_')[0]
    #     if not camera_id in camera_img_dict:
    #         camera_img_dict[camera_id] = []
    width = width // downsample_factor
    height = height // downsample_factor

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, save_fps, (width * 4, height * 3))

    for frame_id in tqdm(frame_list):
        saved_img = np.zeros((height * 3, width * 4, 3)).astype(np.uint8)
        for camera_idx, camera_id in enumerate(camera_list):
            img_path = os.path.join(img_dir, camera_id, camera_id+'_'+frame_id+'.png')
            # assert os.path.exists(img_path)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, height))
                cv2.putText(img, f'{frame_id} {camera_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            else:
                img = np.full((height, width, 3), 255, dtype=np.uint8)
            saved_img[height*(camera_idx//4) : height*((camera_idx//4)+1), width*(camera_idx%4) : width*((camera_idx%4)+1)] = img
        videoWriter.write(saved_img)
    videoWriter.release()

def load_mmpose_joints(root, video_id, camera_list, frame_id, is_righthand):
    joints = []
    for camera_id in camera_list:
        if is_righthand:
            joints_path = os.path.join(root, video_id, 'mmpose_right_hand','predictions', camera_id + '_' + frame_id + '.json')
        else:
            joints_path = os.path.join(root, video_id, 'mmpose_left_hand','predictions', camera_id + '_' + frame_id + '.json')
        assert os.path.exists(joints_path)
        with open(joints_path, 'r') as f:
            joints_info = json.load(f)

        h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info(video_id, camera_id, frame_id, is_righthand)

        # 注意这里是[w_min, h_min]
        joints.append(torch.stack([torch.FloatTensor(x) + torch.FloatTensor([w_min, h_min]) for x in joints_info[0]['keypoints']], dim=0))
    joints = torch.stack(joints, dim=0)
    return joints

def load_mmpose_joints_test(camera_list, frame_id, is_righthand):
    joints_path_prefix = '/home/hlyang/HOI/mmpose/test/output_both_hands/predictions'

    joints = []
    for camera_id in camera_list:
        if is_righthand:
            joints_path = os.path.join(joints_path_prefix, camera_id + '_' + frame_id + '_crop_no_lefthand.json')
        else:
            joints_path = os.path.join(joints_path_prefix, camera_id + '_' + frame_id + '_crop_no_righthand.json')
        assert os.path.exists(joints_path)
        with open(joints_path, 'r') as f:
            joints_info = json.load(f)

        h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info(camera_id, frame_id, is_righthand)

        # 注意这里是[w_min, h_min]
        joints.append(torch.stack([torch.FloatTensor(x) + torch.FloatTensor([w_min, h_min]) for x in joints_info[0]['keypoints']], dim=0))
    joints = torch.stack(joints, dim=0)
    return joints

def load_mmpose_joints_batch(root, video_id, camera_list, frame_list, is_righthand):
    joints = []
    for frame_id in frame_list:
        joints_frame = []
        for camera_id in camera_list:
            if is_righthand:
                joints_path = os.path.join(root, video_id, 'mmpose', 'right_hand','predictions', camera_id + '_' + frame_id + '.json')
            else:
                joints_path = os.path.join(root, video_id, 'mmpose', 'left_hand','predictions', camera_id + '_' + frame_id + '.json')
            assert os.path.exists(joints_path)
            with open(joints_path, 'r') as f:  # TODO: 如果joints_path不存在(或者mmpose没预测出来), 构建一个格式相同的joints_info, 把数值都填成1e9
                joints_info = json.load(f)

            h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info(video_id, camera_id, frame_id, is_righthand)
            # 注意这里是[w_min, h_min]
            joints_frame.append(torch.stack([torch.FloatTensor(x) + torch.FloatTensor([w_min, h_min]) for x in joints_info[0]['keypoints']], dim=0))

        joints_frame = torch.stack(joints_frame, dim=0)
        joints.append(joints_frame)
    # print('------------')
    # print(joints)
    return torch.stack(joints, dim=0)

def load_mmpose_joints_batch_v2(root, video_id, camera_list, frame_list, is_righthand):
    '''
    允许一部分mmpose的结果找不到，某一帧可能存在某些视角缺失。
    '''
    num_frame = len(frame_list)
    num_camera = len(camera_list)
    invalid_exists_bool = False
    valid_mask = torch.full((num_frame, num_camera), 1., dtype=torch.float32)

    joints = []
    for frame_idx, frame_id in enumerate(frame_list):
        joints_frame = []
        for camera_idx, camera_id in enumerate(camera_list):
            if is_righthand:
                joints_path = os.path.join(root, video_id, 'mmpose', 'right_hand','predictions', camera_id + '_' + frame_id + '.json')
            else:
                joints_path = os.path.join(root, video_id, 'mmpose', 'left_hand','predictions', camera_id + '_' + frame_id + '.json')

            if os.path.exists(joints_path): # mmpose结果存在
                with open(joints_path, 'r') as f:
                    joints_info = json.load(f)

                h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info(video_id, camera_id, frame_id, is_righthand)
                # 注意这里是[w_min, h_min]
                joints_frame.append(torch.stack([torch.FloatTensor(xy) + torch.FloatTensor([w_min, h_min]) for xy in joints_info[0]['keypoints']], dim=0))
            else: # TODO: 如果joints_path不存在(或者mmpose没预测出来), 构建一个格式相同的joints_info, 把数值都填成1e9
                invalid_exists_bool = True
                valid_mask[frame_idx, camera_idx] = 0.
                joints_frame.append(torch.full((21,2), 1e9, dtype=torch.float32))

        joints_frame = torch.stack(joints_frame, dim=0)
        joints.append(joints_frame)
    return torch.stack(joints, dim=0), invalid_exists_bool, valid_mask

def load_mmpose_joints_batch_v3(root, date, video_id, crop_info_from_exp_name, camera_list, frame_list, represent_frame_id, is_righthand):
    '''
    允许一部分mmpose的结果找不到，某一帧可能存在某些视角缺失。
    '''
    num_frame = len(frame_list)
    num_camera = len(camera_list)
    invalid_exists_bool = False
    valid_mask = torch.full((num_frame, num_camera), 1., dtype=torch.float32)

    crop_info_batch = load_crop_info_v2(root, date, video_id, crop_info_from_exp_name, camera_list, frame_list, is_righthand, represent_frame_id=represent_frame_id)
    joints = []
    for frame_idx, frame_id in enumerate(frame_list):
        joints_frame = []
        for camera_idx, camera_id in enumerate(camera_list):
            if is_righthand:
                joints_path = os.path.join(root, date, video_id, 'mmpose', 'right_hand','predictions', camera_id + '_' + frame_id + '.json')
            else:
                joints_path = os.path.join(root, date, video_id, 'mmpose', 'left_hand','predictions', camera_id + '_' + frame_id + '.json')

            if os.path.exists(joints_path): # mmpose结果存在
                with open(joints_path, 'r') as f:
                    joints_info = json.load(f)

                # h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info_v2(date, video_id, camera_id, frame_id, is_righthand)
                h_min, h_max, w_min, w_max, h_mean, w_mean = crop_info_batch[frame_id][camera_id]
                # 注意这里是[w_min, h_min]
                joints_frame.append(torch.stack([torch.FloatTensor(xy) + torch.FloatTensor([w_min, h_min]) for xy in joints_info[0]['keypoints']], dim=0))
            else: # TODO: 如果joints_path不存在(或者mmpose没预测出来), 构建一个格式相同的joints_info, 把数值都填成1e9
                invalid_exists_bool = True
                valid_mask[frame_idx, camera_idx] = 0.
                joints_frame.append(torch.full((21,2), 1e9, dtype=torch.float32))

        joints_frame = torch.stack(joints_frame, dim=0)
        joints.append(joints_frame)
    return torch.stack(joints, dim=0), invalid_exists_bool, valid_mask

def load_mmpose_joints_batch_v3_acc(root, date, video_id, from_exp_name, camera_list, frame_list, right_hand_bool, BATCH_SIZE = 20, represent_frame_id = None):
    '''
    允许一部分mmpose的结果找不到，某一帧可能存在某些视角缺失。
    '''
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}
        
    invalid_exists_bool = True
    
    hand_joints_2d_gt_batch = []
    mmpose_valid_mask_batch = []
    for represent_frame_id, represented_frame_list in represent_relation.items():
        if right_hand_bool:
            path = os.path.join(root, date, video_id, from_exp_name, 'right_hand','predictions', f'hand_{represent_frame_id}.pkl')
        else:
            path = os.path.join(root, date, video_id, from_exp_name, 'left_hand','predictions', f'hand_{represent_frame_id}.pkl')
        with open(path, 'rb') as f:
            data_batch = pickle.load(f)
        hand_joints_2d_gt, mmpose_valid_mask = data_batch
            
        for frame_id in represented_frame_list:
            idx = int(frame_id) - int(represent_frame_id)
            hand_joints_2d_gt_batch.append(hand_joints_2d_gt[idx])
            mmpose_valid_mask_batch.append(mmpose_valid_mask[idx])
    
    hand_joints_2d_gt_batch = torch.stack(hand_joints_2d_gt_batch)
    mmpose_valid_mask_batch = torch.stack(mmpose_valid_mask_batch)
    
    return hand_joints_2d_gt_batch, mmpose_valid_mask_batch

def load_mmpose_joints_batch_test(camera_list, frame_list, is_righthand):
    joints_path_prefix = '/home/hlyang/HOI/mmpose/test/output_both_hands/predictions'
    joints = []
    for frame_id in frame_list:
        joints_frame = []
        for camera_id in camera_list:
            if is_righthand:
                joints_path = os.path.join(joints_path_prefix, camera_id + '_' + frame_id + '_crop_no_lefthand.json')
            else:
                joints_path = os.path.join(joints_path_prefix, camera_id + '_' + frame_id + '_crop_no_righthand.json')
            assert os.path.exists(joints_path)
            with open(joints_path, 'r') as f:
                joints_info = json.load(f)

            h_min, h_max, w_min, w_max, h_mean, w_mean = load_crop_info(camera_id, frame_id, is_righthand)
            # 注意这里是[w_min, h_min]
            joints_frame.append(torch.stack([torch.FloatTensor(x) + torch.FloatTensor([w_min, h_min]) for x in joints_info[0]['keypoints']], dim=0))
        joints_frame = torch.stack(joints_frame, dim=0)
        joints.append(joints_frame)
    return torch.stack(joints, dim=0)

def seg_downsample_2_set_test(seg, downsample_factor, adjust_factor):
    '''
    将seg转换成set以计算CD Loss，使用downsample来降低集合中元素中个数。

    input:
    seg: shape如[bs, num_camera, h, w]

    return:
    seg_set: shape如[bs, num_camera, num_verts_in_this_camera, 2]
    num_verts_in_this_camera是不定的，所以实际上是个二维list，元素类型为tensor
    '''

    batch_size, num_camera, h, w = seg.shape
    assert downsample_factor >= 0

    seg = torch.nn.functional.interpolate(seg.float(), size=[h // downsample_factor, w // downsample_factor], mode='nearest')

    seg_set = []
    for batch in range(batch_size):
        seg_set_batch = []
        for camera in range(num_camera):
            idx = torch.flip(torch.nonzero(seg[batch, camera]), dims=[-1])
            # 注意第一个相机的分辨率是别的相机的1/2
            if camera == 0:
                seg_set_batch.append(idx * downsample_factor * adjust_factor / 2)
            else:
                seg_set_batch.append(idx * downsample_factor * adjust_factor)
        seg_set.append(seg_set_batch)

    return seg_set

def seg2set(seg, factor):
    '''
    将seg转换成set以计算CD Loss。

    input:
    seg: shape如[bs, num_camera, h, w]

    return:
    seg_set: shape如[bs, num_camera, num_verts_in_this_camera, 2]
    num_verts_in_this_camera是不定的，所以实际上是个二维list，元素类型为tensor
    '''

    batch_size, num_camera, h, w = seg.shape
    assert factor >= 0

    seg_set = []
    for batch in range(batch_size):
        seg_set_batch = []
        for camera in range(num_camera):
            # seg_set_camera = []
            idx = torch.flip(torch.nonzero(seg[batch, camera]), dims=[-1])
            # for x, y in idx[:]:
            #     seg_set_camera.append(torch.FloatTensor([y * factor, x * factor]))
            # seg_set_camera = torch.stack(seg_set_camera)
            seg_set_batch.append(idx * factor)
        seg_set.append(seg_set_batch)

    return seg_set

def load_mano_info_batch(root, video_id: str, from_exp_name: str, frame_list: str, right_hand_bool: bool):
    if right_hand_bool:
        dir = os.path.join(root, video_id, from_exp_name, 'res', 'right_hand')
    else:
        dir = os.path.join(root, video_id, from_exp_name, 'res', 'left_hand')

    hand_trans_batch = []
    hand_pose_batch = []
    mask_batch = []
    for frame_id in frame_list:
        path = os.path.join(dir, 'hand_' + frame_id + '.pkl')
        assert os.path.exists(path), path
        with open(path, 'rb') as f:
            data = pickle.load(f)
        hand_trans = data['hand_trans']
        hand_pose = data['hand_pose']
        mask = data.get('joints_mask', None)
        hand_trans_batch.append(hand_trans)
        hand_pose_batch.append(hand_pose)
        if mask is not None:
            mask_batch.append(mask)
    hand_trans_batch = torch.stack(hand_trans_batch)
    hand_pose_batch = torch.stack(hand_pose_batch)

    if len(mask_batch) == 0:
        return hand_trans_batch, hand_pose_batch, None
    else:
        mask_batch = torch.stack(mask_batch)
        return hand_trans_batch, hand_pose_batch, mask_batch
    
def load_mano_info_batch_acc(root, date, video_id: str, from_exp_name: str, frame_list: str, right_hand_bool: bool, BATCH_SIZE = 20, represent_frame_id = None, first_batch_len = 2):
    if right_hand_bool:
        dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'right_hand')
    else:
        dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'left_hand')
        
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=first_batch_len)
    else:
        represent_relation = {represent_frame_id: frame_list}

    hand_trans_batch = []
    hand_pose_batch = []
    mask_batch = []
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        # print(represent_frame_id, represented_frame_list)
        path = os.path.join(dir, 'hand_' + represent_frame_id + '.pkl')
        with open(path, 'rb') as f:
            data_batch = pickle.load(f)
            
        for frame_id in represented_frame_list:
            data = data_batch[frame_id]
            
            hand_trans = data['hand_trans']
            hand_pose = data['hand_pose']
            mask = data.get('joints_mask', None)
            
            hand_trans_batch.append(hand_trans)
            hand_pose_batch.append(hand_pose)
            if mask is not None:
                mask_batch.append(mask)
                
    hand_trans_batch = torch.stack(hand_trans_batch)
    hand_pose_batch = torch.stack(hand_pose_batch)

    if len(mask_batch) == 0:
        return hand_trans_batch, hand_pose_batch, None
    else:
        mask_batch = torch.stack(mask_batch)
        return hand_trans_batch, hand_pose_batch, mask_batch
    
def load_obj_info_batch_acc(root, date, video_id: str, from_exp_name: str, frame_list: str, load_obj_type: str, BATCH_SIZE = 20, represent_frame_id = None, first_batch_len = 2):
    
    assert load_obj_type in ('tool', 'obj')
    
    if load_obj_type == 'tool':
        dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'tool')
    else:
        dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'obj')
        
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=first_batch_len)
    else:
        represent_relation = {represent_frame_id: frame_list}

    obj_trans_batch = []
    obj_pose_batch = []
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        # print(represent_frame_id, represented_frame_list)
        path = os.path.join(dir, 'obj_' + represent_frame_id + '.pkl')
        with open(path, 'rb') as f:
            data_batch = pickle.load(f)
            
        for frame_id in represented_frame_list:
            data = data_batch[frame_id]
            
            hand_pose = data['obj_pose']
            hand_trans = data['obj_trans']
            
            obj_pose_batch.append(hand_pose)
            obj_trans_batch.append(hand_trans)
                
    obj_pose_batch = torch.stack(obj_pose_batch)
    obj_trans_batch = torch.stack(obj_trans_batch)

    return obj_pose_batch, obj_trans_batch

def get_mano_info_batch_test(mano_info_path_prefix: str, frame_list: str):
    hand_trans_batch = []
    hand_pose_batch = []
    mask_batch = []
    for frame_id in frame_list:
        path = os.path.join(mano_info_path_prefix, 'hand_' + frame_id + '.pkl')
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        hand_trans = data['hand_trans']
        hand_pose = data['hand_pose']
        mask = data.get('joints_mask', None)
        hand_trans_batch.append(hand_trans)
        hand_pose_batch.append(hand_pose)
        if mask is not None:
            mask_batch.append(mask)
    hand_trans_batch = torch.stack(hand_trans_batch)
    hand_pose_batch = torch.stack(hand_pose_batch)

    if len(mask_batch) == 0:
        return hand_trans_batch, hand_pose_batch, None
    else:
        mask_batch = torch.stack(mask_batch)
        return hand_trans_batch, hand_pose_batch, mask_batch

# def liuyun_convert_axangle_to_euler_2():
#     from transforms3d.euler import euler2axangle
#     from transforms3d.axangles import axangle2euler
#     a = np.float32([0.1, 0.2, 0.3])  # axangle
#     ai, aj, ak = axangle2euler(a)  # axes="sxyz"
#     a_euler = np.float32([ai, aj, ak])
#     a_axangle = euler2axangle(ai, aj, ak)
#     assert a == a_axangle
    #nlopt

def load_joint_ransac_batch(root, video_id, frame_list, mask_type, right_hand_bool):
    '''
    从文件中读取joint ransac得到的hand_trans_3d和joints_mask，并且打成batch。

    return hand_trans_3d, hand_joints_mask
    '''
    assert mask_type in ('joints_mask', 'ransac_mask', 'final_mask')

    if right_hand_bool:
        join_ransac_res_dir = os.path.join(root, video_id, 'joint_ransac_every_joint_triangulation', 'res', 'right_hand')
    else:
        join_ransac_res_dir = os.path.join(root, video_id, 'joint_ransac_every_joint_triangulation', 'res', 'left_hand')
    assert os.path.exists(join_ransac_res_dir)

    trans_batch = []
    mask_batch = []
    for frame_id in frame_list:
        path = os.path.join(join_ransac_res_dir, f'hand_{frame_id}.pkl')
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            trans_batch.append(data['joints_trans'])
            mask_batch.append(data[mask_type])
    trans_batch = torch.stack(trans_batch)
    mask_batch = torch.stack(mask_batch)
    return trans_batch, mask_batch

def load_joint_ransac_batch_acc(root, date, video_id, from_exp_name, frame_list, mask_type, right_hand_bool, BATCH_SIZE = 20, represent_frame_id = None):
    '''
    从文件中读取joint ransac得到的hand_trans_3d和joints_mask，并且打成batch。

    return hand_trans_3d, hand_joints_mask
    '''
    assert mask_type in ('joints_mask', 'ransac_mask', 'final_mask')
    
    if right_hand_bool:
        join_ransac_res_dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'right_hand')
    else:
        join_ransac_res_dir = os.path.join(root, date, video_id, from_exp_name, 'res', 'left_hand')

    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}

    trans_batch = []
    mask_batch = []
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        path = os.path.join(join_ransac_res_dir, 'hand_' + represent_frame_id + '.pkl')
        with open(path, 'rb') as f:
            data_batch = pickle.load(f)
        for frame_id in represented_frame_list:
            data = data_batch[frame_id]
            trans_batch.append(data['joints_trans'])
            mask_batch.append(data[mask_type])
    trans_batch = torch.stack(trans_batch)
    mask_batch = torch.stack(mask_batch)
    return trans_batch, mask_batch

def world2camera_batch_cam(verts_world, R, T):
    '''
    TODO:移除fit_hand_model.py中的world2camera_batch_cam
    verts: [bs, num_verts, 3]
    R: [num_cameras, 3, 3]
    T: [num_cameras, 1, 3]

    return:
    verts_camera: [bs, num_cameras, num_verts, 3]
    '''
    # batch_size = verts_world.shape[0]
    num_camera = R.shape[0]
    verts_camera = torch.einsum('cij, bnj -> bcni', R, verts_world)
    verts_camera = verts_camera + T.resize(1, num_camera, 1, 3)
    return verts_camera

def camera2pixel_batch_cam(verts_camera, K):
    '''
    TODO:移除fit_hand_model.py中的camera2pixel_batch_cam
    verts_camera: [bs, num_cameras, num_verts, 3]
    K: [num_cameras, 3, 3]
    '''
    verts_pixel = torch.einsum('cij, bcnj -> bcni', K, verts_camera)
    ret1 = verts_pixel[..., 0] / verts_pixel[..., 2]
    ret2 = verts_pixel[..., 1] / verts_pixel[..., 2]
    verts_pixel = torch.stack([ret1, ret2], dim=-1)
    return verts_pixel

def get_camera_params(calibration_info_path: str, camera_list):
    '''
    TODO:移除fit_hand_model.py中的get_camera_params
    '''

    assert os.path.exists(calibration_info_path)
    with open(calibration_info_path) as f:
        cali_data = json.load(f)

    R_list = []
    R_inverse_list = []
    T_list = []
    K_list = []
    focal_length_list = []
    principal_point_list = []

    for camera_id in camera_list:
        R = torch.tensor(cali_data[camera_id]['R']).reshape(1, 3, 3)
        T = torch.tensor(cali_data[camera_id]['T']).reshape(1, 3)
        K = torch.tensor(cali_data[camera_id]['K']).reshape(1, 3, 3)
        fx = K[0, 0, 0]
        fy = K[0, 1, 1]
        px = K[0, 0, 2]
        py = K[0, 1, 2]

        R_list.append(R)
        R_inverse_list.append(R.inverse())
        T_list.append(T)
        K_list.append(K)
        focal_length_list.append(torch.tensor([fx, fy]).unsqueeze(0))
        principal_point_list.append(torch.tensor([px, py]).unsqueeze(0))

    R = torch.concatenate(R_list, dim=0)
    R_inverse = torch.concatenate(R_inverse_list, dim=0)
    T = torch.concatenate(T_list, dim=0)
    K = torch.concatenate(K_list, dim=0)
    focal_length = torch.concatenate(focal_length_list, dim=0)
    principal_point = torch.concatenate(principal_point_list, dim=0)

    image_size = torch.tensor([3000, 4096]).unsqueeze(0).repeat(len(camera_list), 1)

    return R, R_inverse, T, K, focal_length, principal_point, image_size

def render_from_mano_params(root, video_id, exp_name, camera_list, frame_list, right_hand_bool, device):
    '''
    从一次实验结果(mano params的格式)得到render的结果，最终结果保存在cpu中。

    Warning: 很容易内存爆炸，不应该那么多数据存内存里。
    '''
    hand_trans_batch, hand_pose_batch, _ = load_mano_info_batch(video_id, exp_name, frame_list, right_hand_bool)
    num_frame = len(frame_list)

    device = torch.device(device)

    calibration_info_path = os.path.join(root, video_id, 'src', 'calibration.json')
    assert os.path.join(calibration_info_path)
    R, R_inverse, T, K, focal_length, principal_point, image_size = get_camera_params(calibration_info_path, camera_list)

    use_pca = False
    ncomps = 45
    if right_hand_bool:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx=0)
    else:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx=0)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    camera = PerspectiveCameras(device=device, R=R_inverse, T=T, image_size=image_size, in_ndc=False, focal_length=-focal_length, principal_point=principal_point)
    raster_settings = RasterizationSettings(
        image_size=(3000, 4096),
        blur_radius=0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings), shader=SoftPhongShader(device=device, cameras=camera, lights=lights))
    faces_idx = mano_layer.th_faces.detach().clone().to(device)

    rendered_image_list = []

    for i in tqdm(range(num_frame)):
        hand_pose = hand_pose_batch[i, ...]
        hand_trans = hand_trans_batch[i, ...]

        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.unsqueeze(0)
        verts, _, _ = mano_layer(hand_pose)
        verts = verts.squeeze()
        verts = verts / 1000.0
        verts += hand_trans
        verts = verts.to(device)


        mesh = Meshes(verts=[verts], faces=[faces_idx])
        color = torch.ones(1, verts.size(0), 3, device=device)
        color[:, :, 2] = 255
        mesh.textures = TexturesVertex(verts_features=color)
        mesh = mesh.extend(R.shape[0])

        images = renderer(mesh)[..., :3].squeeze()
        rendered_image_list.append(images.cpu())
    rendered_image_batch = torch.stack(rendered_image_list)

    return rendered_image_batch

def denoise(mask, half_length = 60, threshold = 900, return_rate = False):
    rows, cols = mask.shape
    cnt = np.zeros(mask.shape)
    idx = np.nonzero(mask)

    if return_rate:
        len1 = len(idx[0])

    min_x = np.maximum(0, idx[0] - half_length)
    max_x = np.minimum(rows - 1, idx[0] + half_length + 1)
    min_y = np.maximum(0, idx[1] - half_length)
    max_y = np.minimum(cols - 1, idx[1] + half_length + 1)
    # 能再次批处理优化吗？
    for x, y, x1, x2, y1, y2 in zip(idx[0], idx[1], min_x, max_x, min_y, max_y):
        cnt[x, y] = mask[x1:x2, y1:y2].sum()
    valid = mask & (cnt > threshold)

    if return_rate:
        len2 = valid.sum()
        rate = len2/len1
        return valid, rate
    else:
        return valid

def denoise_mask(mask, half_length = 60, threshold = 900):
    '''
    input:
        mask: shape: [height, width]
    '''
    assert len(mask.shape) == 2

    right_hand_mask = np.where(mask == 1, True, False)
    left_hand_mask = np.where(mask == 2, True, False)
    object1_mask = np.where(mask == 3, True, False)
    object2_mask = np.where(mask == 4, True, False)
    right_hand_mask = denoise(right_hand_mask, half_length, threshold)
    left_hand_mask = denoise(left_hand_mask, half_length, threshold)

    denoised_mask = np.zeros_like(mask).astype(np.uint8)
    denoised_mask[right_hand_mask] = 1
    denoised_mask[left_hand_mask] = 2
    denoised_mask[object1_mask] = 3
    denoised_mask[object2_mask] = 4

    return denoised_mask

def read_init_crop(root, video_id, camera_id, frame_id):
    '''
    第一排为右手，第二排为左手。
    每一排分别是 min_h, max_h, min_w, max_w

    return shape: [2, 4]
    '''
    path = os.path.join(root, video_id, 'src', 'init_crop', f'{camera_id}_{frame_id}.txt')
    assert os.path.exists(path)
    crop_info = np.loadtxt(path).astype(np.uint32)

    return crop_info

def get_obj_mesh_path(root, obj_id: str):
    obj_dir = os.path.join(root, 'object_models_final')
    obj_filenames = os.listdir(obj_dir)
    valid_filename = [filename for filename in obj_filenames if (filename.endswith(f'object{obj_id}') or ((obj_id[0] == '0') and (filename.endswith(f'object{obj_id[1:]}'))))]  # 加上object防止多个匹配结果
    assert len(valid_filename) == 1, f'obj {obj_id} match failed'
    obj_path = None
    for mesh_name in os.listdir(os.path.join(obj_dir, valid_filename[0])):
        # if "cm.obj" in mesh_name:
        if ("m.obj" in mesh_name) and (not "cm.obj" in mesh_name):
            assert obj_path is None
            obj_path = os.path.join(obj_dir, valid_filename[0], mesh_name)
    assert os.path.exists(obj_path)
    return obj_path

def verts_apply_pose_batch(verts, pose_batch):
    rotation_batch = pose_batch[:, :3, :3] # [num_frame, 3, 3]
    translation_batch = pose_batch[:, :3, 3] # [num_frame, 1, 3]

    verts = torch.einsum('fij, vj -> fvi', rotation_batch, verts) # [num_frame, num_verts, 3]
    verts = verts + translation_batch.unsqueeze(1)
    return verts

def load_nokov_objs_mesh(root, video_id, frame_list):
    '''

    return: tool_verts_batch: [num_frame, num_verts, 3]
            tool_faces
            obj_verts_batch: [num_frame, num_verts, 3]
            obj_faces
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir)
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    frame_idx_list = [(int(frame)-1) for frame in frame_list]

    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2

    tool_mesh_path = get_obj_mesh_path(root, obj_id=tool_id)
    tool_mesh = o3d.io.read_triangle_mesh(tool_mesh_path)
    tool_mesh = tool_mesh.simplify_quadric_decimation(2000)
    tool_verts = np.asarray(tool_mesh.vertices)
    # tool_verts = torch.from_numpy(tool_verts / 100.).float()
    tool_verts = torch.from_numpy(tool_verts).float()
    tool_faces = torch.from_numpy(np.asarray(tool_mesh.triangles))

    tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
    assert os.path.exists(tool_pose_path)
    tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))[frame_idx_list, ...]
    tool_verts_batch = verts_apply_pose_batch(tool_verts, tool_pose_batch)

    obj_mesh_path = get_obj_mesh_path(root, obj_id=obj_id)
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_mesh = obj_mesh.simplify_quadric_decimation(2000)
    obj_verts = np.asarray(obj_mesh.vertices)
    # obj_verts = torch.from_numpy(obj_verts / 100.).float()
    obj_verts = torch.from_numpy(obj_verts).float()
    obj_faces = torch.from_numpy(np.asarray(obj_mesh.triangles))

    obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')
    assert os.path.exists(obj_pose_path)
    obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]
    obj_verts_batch = verts_apply_pose_batch(obj_verts, obj_pose_batch)

    return tool_verts_batch, tool_faces, obj_verts_batch, obj_faces

def check_nokov_exists(root, video_id):
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir), original_nokov_data_dir

    nokov_data_filenames = os.listdir(original_nokov_data_dir)
    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2

    try:
        tool_mesh_path = get_obj_mesh_path(root, obj_id=tool_id)
        obj_mesh_path = get_obj_mesh_path(root, obj_id=obj_id)
        tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
        obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')

        if not os.path.exists(tool_pose_path):
            return False
        if not os.path.exists(obj_pose_path):
            return False
    except:
        return False
    return True


def cal_normals(verts, faces):
    assert len(verts.shape) == 3
    bs = verts.shape[0]


    with torch.no_grad():
        verts = verts.detach().clone().cpu()
        verts = torch.unbind(verts, dim=0)
        verts = [tensor.numpy() for tensor in verts]

        faces = faces.detach().clone().cpu().numpy()

        normals_list = []
        for i in range(bs):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts[i])
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            normals = mesh.vertex_normals
            normals_list.append(torch.from_numpy(np.asarray(normals)))

        normals = torch.stack(normals_list, dim=0)

    return normals

def load_nokov_succeed_list(root, date):
    nokov_succeed_record_path = os.path.join(root ,'record', f'{date}_nokov_succeed.txt')
    nokov_succeed_video_list = []
    with open(nokov_succeed_record_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 1:
            nokov_succeed_video_list.append(parts[0])
    
    return nokov_succeed_video_list

def get_valid_video_list(root: str, date: str, consider_pipiline_failed = False, consider_nokov_failed = False, given_list = None, remove_hand = True):
    valid_record_path = os.path.join(root ,'record', f'{date}_valid_video_id.txt')
    assert os.path.exists(valid_record_path), valid_record_path

    with open(valid_record_path, 'r') as f:
        lines = f.readlines()

    valid_video_list = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 1:
            valid_video_list.append(parts[0])

    if remove_hand:
        valid_video_list = [id for id in valid_video_list if 'hand' not in id]

    valid_video_list = list(set(valid_video_list))
    valid_video_list.sort()
    
    if given_list is not None:
        # valid_video_list_ = [id for id in given_list if id in valid_video_list]
        # valid_video_list = valid_video_list_
        valid_video_list = [id for id in valid_video_list if id in given_list]
        valid_video_list.sort()

    if consider_pipiline_failed:
        pipeline_failed_record_path = os.path.join(root ,'record', f'{date}_pipeline_failed.txt')
        if os.path.exists(pipeline_failed_record_path):
            failed_video_list = []
            with open(pipeline_failed_record_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 1:
                    failed_video_list.append(parts[0])
            valid_video_list = [video_id for video_id in valid_video_list if video_id not in failed_video_list]
            valid_video_list.sort()
            
    if consider_nokov_failed:
        nokov_failed_record_path = os.path.join(root ,'record', f'{date}_nokov_failed.txt')
        if os.path.exists(nokov_failed_record_path):
            failed_video_list = []
            with open(nokov_failed_record_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 1:
                    failed_video_list.append(int(parts[0]))
            
            failed_video_list = [f'{date}_{str(i).zfill(3)}' for i in failed_video_list]
        else:
            failed_video_list = []
                
        nokov_succeed_video_list = load_nokov_succeed_list(root, date)
        
        valid_video_list = [video_id for video_id in valid_video_list if video_id not in failed_video_list and video_id in nokov_succeed_video_list]
        valid_video_list.sort()

    return valid_video_list

def get_time_diff(root, date, error_threshold = 16, valid_threshold = 3):
    '''
    仅适用于20230930及之后的time_diff文件
    '''
    
    time_diff_record_root = os.path.join(root, 'record')
    time_diff_data_path = os.path.join(time_diff_record_root, f'{date}_2m1.txt')
    
    if not os.path.exists(time_diff_data_path):
        return {}
    
    time_diff_data = {}
    with open(time_diff_data_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            time_diff_data[parts[1]] = int(parts[0])
            
    # denoise
    time_diff_array = np.array([v for k, v in time_diff_data.items()])
    invalid_time_diff_list = []
    for time_diff in time_diff_array:
        cnt = np.sum((time_diff_array >= time_diff - error_threshold) & (time_diff_array <= time_diff + error_threshold))
        if cnt <= 3:
            invalid_time_diff_list.append(time_diff)
    keys_to_remove = [k for k, v in time_diff_data.items() if v in invalid_time_diff_list]
    for k in keys_to_remove:
        del time_diff_data[k]
    
    return time_diff_data
    

def get_pipeline_failed_video_list(root, date: str, rename_bool = True):
    peline_failed_record_path = os.path.join(root, 'record', f'{date}_pipeline_failed.txt')
    assert os.path.exists(peline_failed_record_path), peline_failed_record_path

    with open(peline_failed_record_path, 'r') as f:
        lines = f.readlines()

    pipeline_failed_video_list = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 1:
            pipeline_failed_video_list.append(parts[0])


    pipeline_failed_video_list = list(set(pipeline_failed_video_list))
    pipeline_failed_video_list.sort()

    if rename_bool:
        time1 = time()
        peline_failed_record_dst_path = os.path.join(root, 'record', f'{date}_pipeline_failed_{time1}.txt')
        shutil.copy(peline_failed_record_path, peline_failed_record_dst_path)
        os.remove(peline_failed_record_path)

    return pipeline_failed_video_list

def get_num_frame(root, video_id):
    metadata_dir = os.path.join(root, video_id, 'metadata')
    assert os.path.exists(metadata_dir)
    metadata_list = [filename for filename in os.listdir(metadata_dir) if filename.endswith('.pkl')]
    assert len(metadata_list) > 0
    metadata_path = os.path.join(root, video_id, 'metadata', metadata_list[0])

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        num_frame = metadata['num_frame']

    return num_frame

def get_num_frame_v2(root, video_id):
    data = video_id[:8]
    metadata_dir = os.path.join(root, data, video_id, 'metadata')
    assert os.path.exists(metadata_dir), metadata_dir
    metadata_list = [filename for filename in os.listdir(metadata_dir) if filename.endswith('.pkl')]
    assert len(metadata_list) > 0
    metadata_path = os.path.join(root, data, video_id, 'metadata', metadata_list[0])

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        num_frame = metadata['num_frame']

    return num_frame

def load_hand_info_batch_acc(hand_info_dir, frame_list, BATCH_SIZE = 20, represent_frame_id = None):
    '''
    与hoi_io中的load_mano_info_batch功能类似，考虑去重
    '''
    
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}
    
    pose_batch = []
    trans_batch = []
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        path = os.path.join(hand_info_dir, 'hand_' + represent_frame_id + '.pkl')
        with open(path, 'rb') as f:
            data_batch = pickle.load(f)
        
        for frame_id in represented_frame_list:
            data = data_batch[frame_id]
            pose = data['hand_pose']
            trans = data['hand_trans']
            pose_batch.append(pose)
            trans_batch.append(trans)

    pose_batch = torch.stack(pose_batch)
    trans_batch = torch.stack(trans_batch)
    return pose_batch, trans_batch

def load_bg_img_with_resize(root, video_id, camera_id, BATCH_SIZE, frame_list, width = None, height = None):
    
    resize_bool = True if (width is not None and height is not None) else False
    
    date = video_id[:8]
    sub_video_root = os.path.join(root, date, video_id, 'sub_video')
    
    represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list)

    full_img_list = []
    for represent_frame_id, represented_frame_list in represent_relation.items():
        
        video_path = os.path.join(sub_video_root, camera_id, camera_id + '_' + represent_frame_id + '.mp4')

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        suc = cap.isOpened()
        img_list = []
        for i in range(BATCH_SIZE):
            suc, img = cap.read()
            if not suc:
                break
            if resize_bool:
                img = cv2.resize(img, (width, height))
            img_list.append(img)
        
        # num_img = len(img_list)

        img_list_ = []
        for idx in [int(frame) - int(represent_frame_id) for frame in represented_frame_list]:
            # assert idx >= 0 and idx < num_img
            img_list_.append(img_list[idx])
        img_list = img_list_

        for img in img_list:
            full_img_list.append(img)

    return full_img_list

def load_bg_imgs_with_resize(root, video_id, frame_list, camera_list, BATCH_SIZE, width = None, height = None):
    
    rgb_batch = []
    for c_idx, camera in enumerate(camera_list):
        imgs = load_bg_img_with_resize(root, video_id, camera, BATCH_SIZE, frame_list, width, height)
        imgs = np.stack(imgs)
        rgb_batch.append(imgs)
    rgb_batch = np.stack(rgb_batch)
    rgb_batch = np.swapaxes(rgb_batch, 0, 1)
    
    return rgb_batch