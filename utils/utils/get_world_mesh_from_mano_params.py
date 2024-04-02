'''
python utils/get_world_mesh_from_mano_params.py --video_id 20230904_01
'''
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import pickle
from utils.hoi_io import load_mano_info_batch, load_mano_info_batch_acc, cal_represent_frame_list
from manopth.manopth.manolayer import ManoLayer
from tqdm import tqdm
from utils.save_obj import save_obj
import argparse
from utils.scandir import scandir
import torch
import trimesh

def save_world_mesh_from_mano_params(date, video_id, from_exp_name, save_exp_name, frame_list, right_hand_bool):
    num_frame = len(frame_list)

    hand_trans, hand_pose, _ = load_mano_info_batch(video_id, from_exp_name, frame_list, right_hand_bool)

    if right_hand_bool:
        shape_path = os.path.join('/share/hlyang/results', video_id, 'src', 'right_hand_shape.pkl')
    else:
        shape_path = os.path.join('/share/hlyang/results', video_id, 'src', 'left_hand_shape.pkl')

    if os.path.exists(shape_path):
        with open(shape_path, 'rb') as f:
            shape_data = pickle.load(f)
        hand_shape = shape_data['hand_shape']
    else:
        hand_shape = torch.zeros((10))
    hand_shape = hand_shape.unsqueeze(0).expand(num_frame, -1)

    use_pca = False
    ncomps = 45
    if right_hand_bool:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx=0)
    else:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx=0)

    faces_idx = mano_layer.th_faces.detach()

    verts, _, _ = mano_layer(hand_pose, hand_shape)
    verts = verts
    verts = verts / 1000.0
    verts += hand_trans.unsqueeze(1)

    if right_hand_bool:
        save_dir = os.path.join('/share/hlyang/results', video_id, save_exp_name, 'meshes', 'right_hand')
    else:
        save_dir = os.path.join('/share/hlyang/results', video_id, save_exp_name, 'meshes', 'left_hand')
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in tqdm(enumerate(frame_list)):
        save_path = os.path.join(save_dir, f'hand_{frame}.obj')
        save_obj(save_path, verts[i], faces_idx)
        
def save_world_mesh_from_mano_params2(date, video_id, from_exp_name, save_exp_name, frame_list, right_hand_bool):
    num_frame = len(frame_list)

    hand_trans, hand_pose, _ = load_mano_info_batch_acc(date, video_id, from_exp_name, frame_list, right_hand_bool, represent_frame_id=frame_list[0])

    if right_hand_bool:
        shape_path = os.path.join('/share/hlyang/results', date, video_id, 'src', 'right_hand_shape.pkl')
    else:
        shape_path = os.path.join('/share/hlyang/results', date, video_id, 'src', 'left_hand_shape.pkl')

    if os.path.exists(shape_path):
        with open(shape_path, 'rb') as f:
            shape_data = pickle.load(f)
        hand_shape = shape_data['hand_shape']
    else:
        hand_shape = torch.zeros((10))
    hand_shape = hand_shape.unsqueeze(0).expand(num_frame, -1)

    use_pca = False
    ncomps = 45
    if right_hand_bool:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx=0)
    else:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx=0)

    faces_idx = mano_layer.th_faces.detach()

    verts, _, _ = mano_layer(hand_pose, hand_shape)
    verts = verts
    verts = verts / 1000.0
    verts += hand_trans.unsqueeze(1)

    if right_hand_bool:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, save_exp_name, 'meshes', 'right_hand')
    else:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, save_exp_name, 'meshes', 'left_hand')
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in tqdm(enumerate(frame_list)):
        save_path = os.path.join(save_dir, f'hand_{frame}.obj')
        save_obj(save_path, verts[i], faces_idx)

def load_world_meshes_acc(date, video_id, from_exp_name, frame_list, right_hand_bool, BATCH_SIZE = 20, represent_frame_id = None):
    if represent_frame_id is None:
        represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list, first_batch_len=2)
    else:
        represent_relation = {represent_frame_id: frame_list}
        
    trimesh_mesh_dict = {}
    
    if right_hand_bool:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, from_exp_name, 'meshes_batch', 'right_hand')
    else:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, from_exp_name, 'meshes_batch', 'left_hand')
    
    for represent_frame_id, represented_frame_list in represent_relation.items():
        path = os.path.join(save_dir, f'hand_{represent_frame_id}.obj')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        f = data['f']
        for f_idx, frame_id in enumerate(represented_frame_list):
            v = data[frame_id]
            mesh = trimesh.Trimesh(vertices=v, faces=f)
            trimesh_mesh_dict[frame_id] = mesh
    
    return trimesh_mesh_dict
        
    
        
def save_world_mesh_from_mano_params_acc(date, video_id, from_exp_name, save_exp_name, frame_list, represent_frame_id, right_hand_bool):
    num_frame = len(frame_list)

    hand_trans, hand_pose, _ = load_mano_info_batch_acc(date, video_id, from_exp_name, frame_list, right_hand_bool)

    if right_hand_bool:
        shape_path = os.path.join('/share/hlyang/results', date, video_id, 'src', 'right_hand_shape.pkl')
    else:
        shape_path = os.path.join('/share/hlyang/results', date, video_id, 'src', 'left_hand_shape.pkl')

    if os.path.exists(shape_path):
        with open(shape_path, 'rb') as f:
            shape_data = pickle.load(f)
        hand_shape = shape_data['hand_shape']
    else:
        hand_shape = torch.zeros((10))
    hand_shape = hand_shape.unsqueeze(0).expand(num_frame, -1)

    use_pca = False
    ncomps = 45
    if right_hand_bool:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx=0)
    else:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx=0)

    faces_idx = mano_layer.th_faces.detach()

    verts, _, _ = mano_layer(hand_pose, hand_shape)
    verts = verts
    verts = verts / 1000.0
    verts += hand_trans.unsqueeze(1)

    if right_hand_bool:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, save_exp_name, 'meshes_batch', 'right_hand')
    else:
        save_dir = os.path.join('/share/hlyang/results', date, video_id, save_exp_name, 'meshes_batch', 'left_hand')
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {}
    save_data['f'] = faces_idx.numpy()
    for i, frame in tqdm(enumerate(frame_list)):
        save_data[frame] = verts[i].numpy()
        
    save_path = os.path.join(save_dir, f'hand_{represent_frame_id}.obj')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

if __name__ == '__main__':
    exp_name = 'get_world_mesh_from_mano_params_joint'
    # video_id_list = [f'20230904_{str(i).zfill(2)}' for i in range(2,3)]
    # video_id_list = ['20230904_01']

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    # parser.add_argument('--start', required=False, type=int, default=0)
    # parser.add_argument('--end', required=False, type=int, default=0)
    args = parser.parse_args()
    video_id = args.video_id

    # # start和end要么同时输入，要么使用默认，从metadata中取数
    # start = args.start
    # end = args.end
    # assert (start == 0 and end == 0) or (start > 0 and start <= end)
    # if start == 0 and end == 0:
    #     start = 1
    #     metadata_path = os.path.join('/share/hlyang/results', video_id, 'metadata', f'{camera_list[0]}.pkl')
    #     assert os.path.exists(metadata_path)
    #     with open(metadata_path, 'rb') as f:
    #         metadata = pickle.load(f)
    #         end = metadata['num_frame']

    # for video_id in video_id_list:

    # frame_list = [str(frame).zfill(5) for frame in range(start, end + 1)]

    hand_info_exp_name = 'joint_opt'

    res_dir = os.path.join('/share/hlyang/results', video_id, hand_info_exp_name, 'res', 'right_hand')
    fn_list = list(scandir(res_dir, '.pkl'))
    frame_list = [fn.split('_')[1].split('.')[0] for fn in fn_list]
    frame_list.sort()

    save_world_mesh_from_mano_params(video_id, hand_info_exp_name, exp_name, frame_list, right_hand_bool=True)

    res_dir = os.path.join('/share/hlyang/results', video_id, hand_info_exp_name, 'res', 'left_hand')
    fn_list = list(scandir(res_dir, '.pkl'))
    frame_list = [fn.split('_')[1].split('.')[0] for fn in fn_list]
    frame_list.sort()

    save_world_mesh_from_mano_params(video_id, hand_info_exp_name, exp_name, frame_list, right_hand_bool=False)