import os
import sys
sys.path.append('.')
from shutil import copy
from os.path import join
from tqdm import tqdm
from utils.organize_dataset import load_sequence_names_from_organized_record

if __name__ == '__main__':
    root = '/data3/hlyang/results'
    organized_dataset_record_path = '/data3/hlyang/results/dataset/organized_record.txt'
    
    # video_list = [f'20230928_{str(i).zfill(3)}' for i in (20, 23, 26, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 52)]
    
    date = '20231027'
    # video_list = get_valid_video_list(root, date, remove_hand=True, consider_nokov_failed=True, consider_pipiline_failed=True)
    video_list = load_sequence_names_from_organized_record(organized_dataset_record_path, date)
    
    save_exp_name = '4_mesh_vis_addition_test_3'
    video_name_suffix = '_30fps'
    dst_dir = os.path.join(root, 'vis_dataset_test', save_exp_name)
    os.makedirs(dst_dir, exist_ok=True)
    
    for video_id in tqdm(video_list):
        src_path = join(root, date, video_id, save_exp_name, f'{video_id}{video_name_suffix}.mp4')
        if not os.path.exists(src_path):
            print(f'Not exist: {src_path}')
            continue
        dst_path = join(dst_dir, f'{video_id}{video_name_suffix}.mp4')
        copy(src_path, dst_path)