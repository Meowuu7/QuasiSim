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
from utils.hoi_io2 import get_valid_video_list, get_time_diff
from time import time

def is_monotonic(seq):
    if len(seq) <= 1:
        return True  # 空序列或只有一个元素的序列被认为是单调递增的

    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            return False

    return True

def cal_common_timestamps(timestamps_list, error_threshold=16):
    
    # TODO 各个视角视角戳数量一致，就直接算平均数作为common_ts，并且短路
    timestamps_list = [np.array(timestamps) for timestamps in timestamps_list]

    common_timestamps = timestamps_list[0]
    for t_idx, timestamps in enumerate(timestamps_list[1:]):
        common_timestamps_ = []
        for timestamp in timestamps:
            condition = (common_timestamps >= timestamp - error_threshold) & (common_timestamps <= timestamp + error_threshold)
            within_range = common_timestamps[condition]

            if len(within_range) == 1: # 匹配上了
                res = within_range[0]
                # 做个平滑
                modified_cm_ts = (timestamp + res) // 2
                common_timestamps_.append(modified_cm_ts)
            elif len(within_range) == 0: # 没匹配上
                continue
            else: # ？？？
                # print(camera_list[t_idx + 1], within_range)
                res = within_range[np.abs(within_range - timestamp).argmin()]
                modified_cm_ts = (timestamp + res) // 2
                common_timestamps_.append(modified_cm_ts)
                # raise ValueError(f'len(within_range) should be 0 or 1, but got {len(within_range)}')

        common_timestamps = np.array(common_timestamps_)

    return common_timestamps.tolist()
            
def get_reasonable_time_diff(video_id, time_diff_data):
    # if video_id in time_diff_data.keys():
    #     return time_diff_data[video_id]
    # else:
    video_id_list_with_time_diff = list(time_diff_data.keys())
    
    # 如果就在数据中
    if video_id in video_id_list_with_time_diff:
        return time_diff_data[video_id]
    
    # 如果不在数据中，那就匹配
    if video_id[-3:].isdigit():
        abs_arr = np.array([abs(int(video_id[-3:]) - int(k[-3:])) for k in video_id_list_with_time_diff if k[-3:].isdigit()])
        min_idx = np.argmin(abs_arr)
        reasonable_video_id = video_id_list_with_time_diff[min_idx]
    else: # 人手怎么办，暂定是用第一个
        reasonable_video_id = video_id_list_with_time_diff[0]
        
    return time_diff_data[reasonable_video_id]

def modify_timestamps(video_id, time_diff_data, timestamps_list, computer2_ids):
    modified_timestamps_list = []
    
    time_diff = get_reasonable_time_diff(video_id, time_diff_data)
    
    for idx, timestamps in enumerate(timestamps_list):
        if idx in computer2_ids:
            modified_timestamps = [timestamp - time_diff for timestamp in timestamps]
            modified_timestamps_list.append(modified_timestamps)
        else:
            modified_timestamps_list.append(timestamps)
            
    return modified_timestamps_list

def process_frame_loss2(camera_list, upload_date_root, save_root, time_diff_data, video_id, error_threshold):

    date = video_id[:8]

    assert os.path.exists(upload_date_root)
    video_dir = os.path.join(upload_date_root, video_id, 'rgb')
    assert os.path.exists(video_dir), video_dir

    # check if files exist
    for camera_id in camera_list:
        timestamp_path = os.path.join(video_dir, camera_id + '_FrameTimeStamp.txt')
        video_path =  os.path.join(video_dir, camera_id + '.mp4')
        if not os.path.exists(timestamp_path) or not os.path.exists(video_path):
            date = video_id[:8]
            valid_record_path = os.path.join(save_root, 'reference_record', f'{date}_invalid_video_id.txt')
            with open(valid_record_path, 'a') as f:
                f.write(f'{video_id}\n')
            return

    computer1_camera_list = ['21218078', '22139906', '22139908', '22139910', '22139911', '22139913', '22139914', '22139946']
    computer2_camera_list = [camera for camera in camera_list if camera not in computer1_camera_list]
    assert len(computer2_camera_list) == 4
    computer1_ids = [camera_list.index(camera) for camera in computer1_camera_list]
    computer2_ids = [camera_list.index(camera) for camera in computer2_camera_list]

    timestamps_list = []
    for camera_id in camera_list:
        timestamp_path = os.path.join(video_dir, camera_id + '_FrameTimeStamp.txt')

        with open(timestamp_path, 'r') as f:
            lines = f.readlines()

        timestamps = []
        cnt = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                timestamp = parts[1]
                timestamps.append(int(timestamp))
                cnt += 1

        timestamps_list.append(timestamps)

    # # 使用计算出的误差
    # modified_timestamps_list = []
    # for idx, timestamps in enumerate(timestamps_list):
    #     if idx in computer2_ids:
    #         modified_timestamps = [timestamp - time_diff for timestamp in timestamps]
    #         modified_timestamps_list.append(modified_timestamps)
    #     else:
    #         modified_timestamps_list.append(timestamps)
    
    modified_timestamps_list = modify_timestamps(video_id, time_diff_data, timestamps_list, computer2_ids)
    # for mtsl in modified_timestamps_list:
    #     print(mtsl[0])

    common_timestamps = cal_common_timestamps(modified_timestamps_list, error_threshold)
    num_common_frame = len(common_timestamps)

    if num_common_frame <= 0:
        match_failed_record_path = os.path.join(save_root, 'reference_record', f'{date}_match_failed_video_id.txt')
        with open(match_failed_record_path, 'a') as f:
            f.write(f'{video_id}\n')
        return

    valid_record_path = os.path.join(save_root, 'record', f'{date}_valid_video_id.txt')
    with open(valid_record_path, 'a') as f:
        f.write(f'{video_id}\n')

    result_dir = os.path.join(save_root, date, video_id)
    
    os.makedirs(result_dir, exist_ok=True)
    metadata_dir = os.path.join(result_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    common_frame_record_path = os.path.join(metadata_dir, 'common_timestamp.txt')
    with open(common_frame_record_path, 'w') as f:
        for timestamp in common_timestamps:
            f.write(f'{timestamp}\n')

    line = f'{video_id}: {num_common_frame} '
    for idx, camera_id in enumerate(camera_list):
        cnt2frame_id_dict = {}
        frame_id2cnt_dict = {}
        cnt_list = []
        for idx2, cm_ts in enumerate(common_timestamps):
            for original_ts_idx, timestamp in enumerate(modified_timestamps_list[idx]):
                if abs(cm_ts - timestamp) <= error_threshold:
                    idx_in_original_ts_list = original_ts_idx
                    break

            # frame_id从00001开始记录
            cnt = idx_in_original_ts_list + 1
            frame_id = str(idx2 + 1).zfill(5)
            cnt_list.append(cnt)
            cnt2frame_id_dict[cnt] = frame_id
            frame_id2cnt_dict[frame_id] = cnt

        assert len(cnt_list) == num_common_frame
        # print(camera_id, len(modified_timestamps_list[idx]), cnt2frame_id_dict)
        metadata = {'num_frame': num_common_frame, 'original_num_frame': len(modified_timestamps_list[idx]), 'cnt_list':cnt_list, 'cnt2frame_id_dict': cnt2frame_id_dict, 'frame_id2cnt_dict': frame_id2cnt_dict}

        metadata_path = os.path.join(metadata_dir, camera_id+'.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        line += f'{camera_id}: {len(modified_timestamps_list[idx])} '
    line += '\n'
    match_record_path = os.path.join(save_root, 'reference_record', f'{date}_match_record.txt')
    with open(match_record_path, 'a') as f:
        f.write(line)

if __name__ == '__main__':

    # camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']

    date = '20230930'
    upload_root_dir = '/data2/HOI-mocap'
    upload_date_root = os.path.join(upload_root_dir, date)

    save_root = '/data2/hlyang/results'
    save_date_root = os.path.join(save_root, date)
    os.makedirs(save_date_root, exist_ok=True)

    dir_list = os.listdir(upload_date_root)
    video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]

    video_list.sort()
    
    print(video_list)

    error_threshold = 17

    time_diff_data = get_time_diff(save_root, date)

    for video_id in tqdm(video_list):
        try:
            process_frame_loss2(camera_list, upload_date_root, save_root, time_diff_data, video_id, error_threshold)
            
        except Exception as error:
            traceback.print_exc()
            print(error)
            continue

    video_list = get_valid_video_list(save_root, date, remove_hand=False)
    for video_id in tqdm(video_list):
        # cp calibration file
        src_cali_path = os.path.join(upload_date_root, video_id, 'src', 'calibration.json')
        dst_src_dir = os.path.join(save_date_root, video_id, 'src')
        os.makedirs(dst_src_dir, exist_ok=True)
        dst_cali_path = os.path.join(dst_src_dir, 'calibration.json')
        copy(src_cali_path, dst_cali_path)