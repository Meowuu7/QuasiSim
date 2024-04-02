import os
import sys
sys.path.append('.')
import pickle
import numpy as np
import open3d as o3d
import torch
import json
from tqdm import tqdm
import traceback

from manopth.manopth.manolayer import ManoLayer
from utils.hoi_io2 import get_num_frame_v2
from utils.organize_dataset import load_sequence_names_from_organized_record, txt2intrinsic, load_simplied_nokov_objs_mesh, load_organized_mano_info, load_dates_from_organized_record

def load_simplied_objs_template(root, video_id, frame_list = None, use_cm = True):
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

if __name__ == '__main__':
    # date_list = ['20230930']
    device = 'cuda:0'

    root = '/data3/hlyang/results'
    upload_root = '/data3/hlyang/HOI-mocap'
    nokov_root = upload_root
    dataset_root = os.path.join(root, 'dataset')
    hand_pose_organized_record_path = os.path.join(dataset_root, 'organized_record.txt')
    
    date_list = load_dates_from_organized_record(hand_pose_organized_record_path)
    
    save_root = '/data3/hlyang/results/test_data'

    for date in date_list:
        video_list = load_sequence_names_from_organized_record(hand_pose_organized_record_path, date)

        save_dir = os.path.join(save_root, date)
        os.makedirs(save_dir, exist_ok=True)

        for video_id in tqdm(video_list):
            try:
                _, _, obj_verts, obj_faces = load_simplied_objs_template(nokov_root, video_id, use_cm = True)
                obj_verts = obj_verts.numpy()
                obj_faces = obj_faces.numpy()
                obj_mesh = o3d.geometry.TriangleMesh()
                obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
                obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
                
                save_path = os.path.join(save_dir, f'{video_id}.obj')
                o3d.io.write_triangle_mesh(save_path, obj_mesh)
            
            except Exception as err:
                traceback.print_exc()
                print(err)
                continue