import os
import sys
sys.path.append('.')
import pickle
import numpy as np
# import open3d as o3d
import torch
# import json
from tqdm import tqdm
import traceback

from manopth.manolayer import ManoLayer
# from utils.hoi_io2 import get_num_frame_v2
from utils.organize_dataset import load_sequence_names_from_organized_record, txt2intrinsic, load_simplied_nokov_objs_mesh, load_organized_mano_info, load_dates_from_organized_record

def load_objs_orientation(root, video_id, frame_list = None):
    '''

    return: tool_pose_batch, obj_pose_batch
    '''
    date = video_id[:8]
    original_nokov_data_dir = os.path.join(root, date, video_id, 'nokov')
    assert os.path.exists(original_nokov_data_dir)
    nokov_data_filenames = os.listdir(original_nokov_data_dir)

    if frame_list is not None:
        frame_idx_list = [(int(frame)-1) for frame in frame_list]

    tool_id = nokov_data_filenames[0].split("_")[1] # obj1
    obj_id = nokov_data_filenames[0].split("_")[2] # obj2

    tool_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{tool_id}.npy')
    assert os.path.exists(tool_pose_path)
    if frame_list is None:
        tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))
    else:
        tool_pose_batch = torch.from_numpy(np.load(tool_pose_path))[frame_idx_list, ...]

    obj_pose_path = os.path.join(root, 'HO_poses', date, video_id, 'objpose', f'{obj_id}.npy')
    assert os.path.exists(obj_pose_path)
    if frame_list is None:
        obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))
    else:
        obj_pose_batch = torch.from_numpy(np.load(obj_pose_path))[frame_idx_list, ...]

    return tool_pose_batch, obj_pose_batch

if __name__ == '__main__':
    # date_list = ['20230930']
    device = 'cuda:0'

    root = '/data3/hlyang/results'
    upload_root = '/data3/hlyang/HOI-mocap'
    nokov_root = upload_root
    dataset_root = os.path.join(root, 'dataset')
    hand_pose_organized_record_path = os.path.join(dataset_root, 'organized_record.txt')
    
    date_list = load_dates_from_organized_record(hand_pose_organized_record_path)
    
    # save_root = '/data3/hlyang/results/test_data'
    save_root = '/data3/datasets/xueyi/taco/processed_data'
    
    mano_path = "/data1/xueyi/mano_models/mano/models"

    for date in date_list:
        video_list = load_sequence_names_from_organized_record(hand_pose_organized_record_path, date)

        for video_id in tqdm(video_list):
            try:
                save_dir = os.path.join(save_root, date)
                os.makedirs(save_dir, exist_ok=True)

                (right_hand_pose, right_hand_trans, right_hand_shape) = load_organized_mano_info(dataset_root, date, video_id, frame_list = None, right_hand_bool=True)
                (left_hand_pose, left_hand_trans, left_hand_shape) = load_organized_mano_info(dataset_root, date, video_id, frame_list = None, right_hand_bool=False)
                tool_verts_batch, tool_faces, obj_verts_batch, obj_faces = load_simplied_nokov_objs_mesh(nokov_root, video_id, frame_list = None, use_cm = True)
                tool_pose_batch, obj_pose_batch = load_objs_orientation(nokov_root, video_id, frame_list = None)
                
                batch_size = right_hand_pose.shape[0] 
                
                if right_hand_shape is not None and left_hand_shape is not None:
                    use_shape = True
                else:
                    use_shape = False
                
                right_hand_pose = right_hand_pose.to(device)
                right_hand_trans = right_hand_trans.to(device)
                left_hand_pose = left_hand_pose.to(device)
                left_hand_trans = left_hand_trans.to(device)
                tool_verts_batch = tool_verts_batch.to(device)
                obj_verts_batch = obj_verts_batch.to(device)
                
                if use_shape:
                    right_hand_shape = right_hand_shape.to(device)
                    left_hand_shape = left_hand_shape.to(device)        
                
                # get hand meshes
                use_pca = False
                ncomps = 45
                left_hand_mano_layer = ManoLayer(mano_root=mano_path, use_pca=use_pca, ncomps=ncomps, side='left', center_idx = 0).to(device)
                right_hand_mano_layer = ManoLayer(mano_root=mano_path, use_pca=use_pca, ncomps=ncomps, side='right', center_idx = 0).to(device)
                
                with torch.no_grad():
                    if use_shape:
                        right_hand_verts, _ = right_hand_mano_layer(right_hand_pose, right_hand_shape.unsqueeze(0).expand(batch_size, -1))
                    else:
                        right_hand_verts, _ = right_hand_mano_layer(right_hand_pose)
                    right_hand_verts = right_hand_verts
                    right_hand_verts = right_hand_verts / 1000.0
                    right_hand_verts += right_hand_trans.unsqueeze(1)
                    
                    if use_shape:
                        left_hand_verts, _ = left_hand_mano_layer(left_hand_pose, left_hand_shape.unsqueeze(0).expand(batch_size, -1))
                    else:
                        left_hand_verts, _ = left_hand_mano_layer(left_hand_pose)
                    left_hand_verts = left_hand_verts
                    left_hand_verts = left_hand_verts / 1000.0
                    left_hand_verts += left_hand_trans.unsqueeze(1)
                    
                # left hand
                hand_pose = left_hand_pose.cpu().numpy()
                hand_trans = left_hand_trans.cpu().numpy()
                if use_shape:
                    hand_shape = left_hand_shape.cpu().numpy()
                else:
                    hand_shape = torch.zeros(10).cpu().numpy()
                hand_verts = left_hand_verts.cpu().numpy()
                hand_faces = left_hand_mano_layer.th_faces.cpu().numpy()
                obj_verts = obj_verts_batch.cpu().numpy()
                obj_faces = obj_faces.cpu().numpy()
                
                # 
                rgt_hand_pose = right_hand_pose.cpu().numpy()
                rgt_hand_trans = right_hand_trans.cpu().numpy()
                if use_shape:
                    rgt_hand_shape = right_hand_shape.cpu().numpy()
                else:
                    rgt_hand_shape = torch.zeros(10).cpu().numpy()
                rgt_hand_verts = right_hand_verts.cpu().numpy()
                rgt_hand_faces = right_hand_mano_layer.th_faces.cpu().numpy()
                rgt_obj_verts = tool_verts_batch.cpu().numpy()
                rgt_obj_faces = tool_faces.cpu().numpy()
                
                
                # print(hand_verts.shape, hand_faces.shape, obj_verts.shape, obj_faces.shape)
                save_path = os.path.join(save_dir, f'right_{video_id}.pkl')
                # save_data = {
                #     'hand_pose': hand_pose, 'hand_trans': hand_trans, 'hand_shape': hand_shape, 'hand_verts': hand_verts, 'hand_faces': hand_faces, 'obj_verts': obj_verts, 'obj_faces': obj_faces, 'obj_pose': obj_pose_batch.cpu().numpy(),
                    
                # }
                save_data = {
                    'hand_pose': rgt_hand_pose, 'hand_trans': rgt_hand_trans, 'hand_shape': rgt_hand_shape, 'hand_verts': rgt_hand_verts, 'hand_faces': rgt_hand_faces, 'obj_verts': rgt_obj_verts, 'obj_faces': rgt_obj_faces, 'obj_pose': tool_pose_batch.cpu().numpy(),
                    
                }
                # print(save_data['hand_pose'].shape, save_data['hand_trans'].shape, save_data['hand_shape'].shape, save_data['hand_verts'].shape, save_data['hand_faces'].shape, save_data['obj_verts'].shape, save_data['obj_faces'].shape, save_data['obj_pose'].shape)
                pickle.dump(save_data, open(save_path, 'wb'))
            
            except Exception as err:
                traceback.print_exc()
                print(err)
                continue