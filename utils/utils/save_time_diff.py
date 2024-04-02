'''
解析录制时的{camera_id}__FrameTimeStamp.txt，根据timestamp丢弃录制时有丢失的帧，并将剩余的帧数据存为pkl。之后video2img.py等一系列程序将基于该pkl进行处理。

在/share/hlyang/results/record生成{date}_invalid_video_id.txt, {date}_match_failed_video_id.txt, {date}_valid_video_id.txt等

example:
python utils/process_frame_loss.py --root_dir /share/datasets/HOI-mocap/20230904 --video_id 20230904_01
'''
import traceback

import os
import sys
sys.path.append('.')
import argparse
import pickle
from shutil import copy
import numpy as np
from tqdm import tqdm
from time import time
from loguru import logger

from utils.hoi_io2 import get_valid_video_list, get_time_diff
from utils.process_frame_loss import get_computer_time_diff
from utils.process_frame_loss2 import get_reasonable_time_diff
from utils.organize_dataset import load_sequence_names_from_organized_record, txt2intrinsic, load_simplied_nokov_objs_mesh, load_organized_mano_info, load_dates_from_organized_record


if __name__ == '__main__':

    root = '/data3/hlyang/results'
    dataset_root = os.path.join(root, 'dataset')
    hand_pose_organized_record_path = os.path.join(dataset_root, 'organized_record.txt')
    date_list = load_dates_from_organized_record(hand_pose_organized_record_path)
    new_form_date_list = ['20230919', '20230930', '20231005', '20231006', '20231010', '20231013', '20231015', '20231019', '20231020', '20231024', '20231026', '20231027', '20231031', '20231102', '20231103', '20231104', '20231105']
    old_form_date_list = [date for date in date_list if date not in new_form_date_list]
    
    save_root = '/data3/hlyang/results'
    
    upload_root_dir = '/data2/HOI-mocap'
    tot_time_diff = {}
    
    for date in date_list:
        upload_date_root = os.path.join(upload_root_dir, date)
        dir_list = os.listdir(upload_date_root)
        video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]
        video_list.sort()
        
        if date in new_form_date_list:
            time_diff_data = get_time_diff(save_root, date)

            pbar = tqdm(video_list)
            for video_id in pbar:
                try:
                    pbar.set_description(f'Processing {video_id}')
                    time_diff = get_reasonable_time_diff(video_id, time_diff_data)
                    tot_time_diff[video_id] = time_diff
                    
                except Exception as error:
                    traceback.print_exc()
                    print(error)
                    continue
        
        else:
            error_threshold = 17
            time_diff = get_computer_time_diff(date, error_threshold)
            pbar = tqdm(video_list)
            for video_id in pbar:
                try:
                    pbar.set_description(f'Processing {video_id}')
                    tot_time_diff[video_id] = time_diff
                    
                except Exception as error:
                    traceback.print_exc()
                    print(error)
                    continue
            
            
    save_path = os.path.join(save_root, 'dataset', 'time_diff.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(tot_time_diff, f)
