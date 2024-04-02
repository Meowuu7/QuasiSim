import os
import sys
sys.path.append('.')
import cv2
import numpy as np
import pickle
from os.path import join
from transforms3d.quaternions import quat2mat, mat2quat
# from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import torch
import open3d as o3d

from utils.hoi_io2 import verts_apply_pose_batch

def organize_record_file(path):
    video_id_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 1:
            video_id_list.append(parts[0])

    video_id_list = list(set(video_id_list))
    video_id_list.sort(key=lambda x:int(x))

    with open(path, 'w') as f:
        for id in video_id_list:
            f.write(f'{id}\n')

def add_a_line(path, line):
    with open(path, 'a') as f:
        f.write(f'{line}\n')

def mp42imgs(video_path, return_rgb = True, max_cnt=None, width = None, height = None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    imgs = []

    resize_bool = True if (width is not None and height is not None) else False

    suc = cap.isOpened()
    cnt = 1 # 原视频第{n}帧，从1开始计数
    while True:
        suc, img = cap.read()
        if not suc:
            break
        if return_rgb:
            if resize_bool:
                imgs.append(cv2.resize(img, (width, height)))
            else:
                imgs.append(img)
        cnt += 1
        if (not max_cnt is None) and (cnt > max_cnt): # 最多取到第max_cnt帧
            break

    cap.release()

    if return_rgb:
        return imgs, fps, W, H, cnt - 1
    else:
        return None, fps, W, H, cnt - 1

def txt2intrinsic(txt_path):
    intrin = np.eye(3)
    dist = np.zeros(5)
    cnt = -1
    with open(txt_path, 'r') as f:
        for line in f:
            cnt += 1
            line = line.strip().split(',')
            values = np.float32([float(v) for v in line])
            if cnt <= 2:
                intrin[cnt] = values
            else:
                dist = values
    return intrin, dist

def get_ego_rigid_xrs_path(NOKOV_data_dir):
    xrs_path = None
    for fn in os.listdir(NOKOV_data_dir):
        if (fn[-4:] == ".xrs") and ("helmet" in fn):
            assert xrs_path is None
            xrs_path = join(NOKOV_data_dir, fn)
    return xrs_path

def parse_xrs(xrs_paths):

    data_list = []

    for xrs_path in xrs_paths:
        cnt = 0
        data = {
            "poses": [],
        }
        with open(xrs_path, "r") as f:
            for line in f:
                cnt += 1
                line = line.strip()
                if cnt <= 11:
                    continue
                values = line.split("\t")

                assert len(values) == 17

                pose = np.eye(4).astype(np.float32)
                pose[:3, 3] = 10000
                t = np.float32([float(values[2]), float(values[3]), float(values[4])]) / 1000
                pose[:3, 3] = t
                q = np.float32([float(values[8]), float(values[5]), float(values[6]), float(values[7])])  # xyzw -> wxyz
                R = quat2mat(q)
                pose[:3, :3] = R

                data["poses"].append(pose)

        data["poses"] = np.float32(data["poses"])  # (N, 4, 4)

        data_list.append(data)

    return data_list

def load_nokov_objs_params_arctic_style(root, video_id, frame_list):
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

    tool_id = nokov_data_filenames[0].split('_')[1] # obj1
    obj_id = nokov_data_filenames[0].split('_')[2] # obj2

    tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
    assert os.path.exists(tool_pose_path)
    tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))[frame_idx_list, ...]
    tool_rot_batch = tool_pose_batch[:, :3, :3] # [num_frame, 3, 3]
    tool_anxis_angle_batch = matrix_to_axis_angle(tool_rot_batch)  # [num_frame, 3]
    tool_trans_batch = tool_pose_batch[:, :3, 3] # [num_frame, 3]
    tool_params = torch.cat([tool_anxis_angle_batch, tool_trans_batch], dim=1)

    obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')
    assert os.path.exists(obj_pose_path)
    obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]
    obj_rot_batch = obj_pose_batch[:, :3, :3] # [num_frame, 3, 3]
    obj_anxis_angle_batch = matrix_to_axis_angle(obj_rot_batch)  # [num_frame, 3]
    obj_trans_batch = obj_pose_batch[:, :3, 3] # [num_frame, 3]
    obj_params = torch.cat([obj_anxis_angle_batch, obj_trans_batch], dim=1)

    return tool_params.numpy(), obj_params.numpy()

def cvt_mano_info_2_arctic_style(hand_pose, hand_trans, hand_shape):
    num_frame = hand_pose.shape[0]

    data = {}
    data['rot'] = hand_pose[..., :3].numpy()
    data['pose'] = hand_pose[..., 3:].numpy()
    data['trans'] = hand_trans.numpy()
    if hand_shape is not None:
        data['shape'] = hand_shape.numpy()
    else:
        data['shape'] = np.zeros((10))

    # data['fitting_err'] = [0. for i in range(num_frame)]
    data['fitting_err'] = [0. for i in range(num_frame)]
    return data

def load_simplied_nokov_objs_mesh(root, video_id, frame_list = None, use_cm = True):
    '''

    return: tool_verts_batch: [num_frame, num_verts, 3]
            tool_faces
            obj_verts_batch: [num_frame, num_verts, 3]
            obj_faces
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir), original_nokov_data_dir
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    if frame_list is not None:
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
    if frame_list is None:
        tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))
    else:
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
    if frame_list is None:
        obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))
    else:
        obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]
    obj_verts_batch = verts_apply_pose_batch(obj_verts, obj_pose_batch)

    return tool_verts_batch, tool_faces, obj_verts_batch, obj_faces

def load_sequence_names_from_organized_record(path: str, date: str):
    organized_sequence_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0 and parts[0].startswith(date):
            organized_sequence_list.append(parts[0])

    organized_sequence_list = list(set(organized_sequence_list))
    organized_sequence_list.sort(key=lambda x:int(x))

    return organized_sequence_list

def load_dates_from_organized_record(path: str):
    organized_date_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            organized_date_list.append(parts[0][:8])

    organized_date_list = list(set(organized_date_list))
    organized_date_list.sort(key=lambda x:int(x))

    return organized_date_list

def load_organized_mano_info(dataset_root, date, video_id, frame_list = None, mano_dirname = 'mano_wo_contact', right_hand_bool = True):
    '''
    return:(hand_pose, hand_trans, hand_shape)
        hand_pose: torch tensor, shape [num_frame, 48]
        hand_trans: torch tensor, shape [num_frame, 3]
        hand_shape: None or torch tensor, shape [10]
    '''

    if right_hand_bool:
        path = os.path.join(dataset_root, date, video_id, mano_dirname, 'right_hand.pkl')
    else:
        path = os.path.join(dataset_root, date, video_id, mano_dirname, 'left_hand.pkl')

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if frame_list is None:
        frame_list = list(data.keys())
        frame_list.sort()
    hand_pose = []
    hand_trans = []
    for frame in frame_list:
        hand_pose.append(data[frame]['hand_pose'])
        hand_trans.append(data[frame]['hand_trans'])

    hand_pose = torch.stack(hand_pose)
    hand_trans = torch.stack(hand_trans)

    if right_hand_bool:
        shape_path = os.path.join(dataset_root, date, video_id, 'src', 'right_hand_shape.pkl')
    else:
        shape_path = os.path.join(dataset_root, date, video_id, 'src', 'left_hand_shape.pkl')

    if os.path.exists(shape_path):
        with open(shape_path, 'rb') as f:
            hand_shape = pickle.load(f)['hand_shape']
    else:
        hand_shape = None

    return (hand_pose, hand_trans, hand_shape)

def load_zero_nokov_objs_mesh(root, video_id, use_cm = True):
    '''

    return: tool_verts_batch: [num_verts, 3]
            tool_faces
            obj_verts_batch: [num_verts, 3]
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

def load_organized_rgb_batch_with_resize(dataset_root, date, video_id, frame_list, camera_list, width, height):

    frame_idx_list = [(int(frame)-1) for frame in frame_list]

    rgb_batch = []
    for camera in camera_list:
        path = os.path.join(dataset_root, date, video_id, 'rgb', f'{camera}.mp4')
        imgs, _, _, _, cnt  = mp42imgs(path, return_rgb=True, width=width, height=height)
        imgs = np.stack(imgs)
        rgb_batch.append(imgs)

    rgb_batch = np.stack(rgb_batch)
    rgb_batch = rgb_batch[:, frame_idx_list, ...]
    rgb_batch = np.swapaxes(rgb_batch, 0, 1)

    return rgb_batch

def load_organized_ego_rgb_with_resize(dataset_root, date, video_id, frame_list, width, height):

    if frame_list is not None:
        frame_idx_list = [(int(frame)-1) for frame in frame_list]

    path = os.path.join(dataset_root, date, video_id, 'egocentric_rgb.mp4')
    imgs, _, _, _, cnt  = mp42imgs(path, return_rgb=True, width=width, height=height)
    imgs = np.stack(imgs)
    if frame_list is not None:
        imgs = imgs[frame_idx_list, ...]
    
    return imgs

def load_interaction_field(root, date, video_id):
    save_path = os.path.join(root, 'interaction_field', date, f'{video_id}.pkl')
    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    F_right_hand_2_tool = data['F_right_hand_2_tool']
    F_right_hand_2_tool_idxs = data['F_right_hand_2_tool_idxs']
    F_tool_2_right_hand = data['F_tool_2_right_hand']
    F_tool_2_right_hand_idxs = data['F_tool_2_right_hand_idxs']
    F_left_hand_2_obj = data['F_left_hand_2_obj']
    F_left_hand_2_obj_idxs = data['F_left_hand_2_obj_idxs']
    F_obj_2_left_hand = data['F_obj_2_left_hand']
    F_obj_2_left_hand_idxs = data['F_obj_2_left_hand_idxs']
    F_tool_2_obj = data['F_tool_2_obj']
    F_tool_2_obj_idxs = data['F_tool_2_obj_idxs']
    F_obj_2_tool = data['F_obj_2_tool']
    F_obj_2_tool_idxs = data['F_obj_2_tool_idxs']

    return F_right_hand_2_tool, F_right_hand_2_tool_idxs, F_tool_2_right_hand, F_tool_2_right_hand_idxs, F_left_hand_2_obj, F_left_hand_2_obj_idxs, F_obj_2_left_hand, F_obj_2_left_hand_idxs, F_tool_2_obj, F_tool_2_obj_idxs, F_obj_2_tool, F_obj_2_tool_idxs

def cvt_interaction_field_2_colors(interaction_field_list):

    colors = []

    for interaction_field in interaction_field_list:

        device = interaction_field.device
        # interaction_field = interaction_field.exp()
        # interaction_field = torch.pow(1000000.0, interaction_field)
        interaction_field = torch.clip(interaction_field, 0., 0.01)

        # min_weight = interaction_field.min()
        # max_weight = interaction_field.max()
        min_weight = 0.
        max_weight = 0.01

        red = (max_weight - interaction_field) / (max_weight - min_weight)
        blue = (interaction_field - min_weight) / (max_weight - min_weight)
        color = torch.stack((red, red, blue)).transpose(0, 1) # RGB
        # color = torch.stack((red, torch.zeros_like(red, device=device), blue)).transpose(0, 1) # RGB
        # color = torch.stack((blue, torch.zeros_like(red, device=device), red)).transpose(0, 1) # BRG


        colors.append(color)

    return colors

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