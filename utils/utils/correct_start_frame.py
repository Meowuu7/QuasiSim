import os
import sys
sys.path.append('.')
from shutil import copy, rmtree, copytree
from os.path import join
from hoi_io2 import get_valid_video_list
from tqdm import tqdm
import pickle

def load_valid_start_dict(root, date):
    path = os.path.join(root, 'record', f'{date}_valid_start.txt')
    valid_start = {}
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                valid_start[parts[0]] = parts[1]
                
    return valid_start

def correct_metadata(metadata_dir, camera_list, valid_start):
    cm_ts_path = os.path.join(metadata_dir, 'common_timestamp.txt')
    valid_start_int = int(valid_start)
    
    cm_ts_list = []
    with open(cm_ts_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip()
            if len(parts) > 0:
                cm_ts_list.append(line)
    
    with open(cm_ts_path, 'w') as f:
        for line in cm_ts_list[valid_start_int-1:]:
            f.write(line)
            
    for camera in camera_list:
        metadata_path = os.path.join(metadata_dir, f'{camera}.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        metadata['num_frame'] = metadata['num_frame'] - (valid_start_int-1)
        metadata['cnt_list'] = metadata['cnt_list'][valid_start_int-1:]
        
        cnt2frame_id_dict = metadata['cnt2frame_id_dict']
        
        cnt2frame_id_dict_new = {}
        for cnt, frame in cnt2frame_id_dict.items():
            if int(frame) < valid_start_int:
                continue
            else:
                cnt2frame_id_dict_new[cnt] = str(int(frame) - valid_start_int + 1).zfill(5)
        metadata['cnt2frame_id_dict'] = cnt2frame_id_dict_new
        metadata['frame_id2cnt_dict'] = {v: k for k, v in cnt2frame_id_dict_new.items()}
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

if __name__ == '__main__':
    
    root = '/data2/hlyang/results'
    upload_root = '/data2/HOI-mocap'
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']
    date = '20231019'
    
    video_list = get_valid_video_list(root, date, consider_nokov_failed=True)
    # video_list = ['20231019_001']
    valid_start_dict = load_valid_start_dict(root, date)
    
    for video_id in video_list:
        valid_start = valid_start_dict[video_id]
        
        metadata_dir = os.path.join(root, date, video_id, 'metadata')
        metadata_backup_dir = os.path.join(root, date, video_id, 'metadata_backup')
        
        copytree(metadata_dir, metadata_backup_dir)
        
        correct_metadata(metadata_dir, camera_list, valid_start)
    