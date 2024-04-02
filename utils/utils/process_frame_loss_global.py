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
from utils.hoi_io import get_valid_video_list
from time import time

def is_monotonic(seq):
    if len(seq) <= 1:
        return True  # 空序列或只有一个元素的序列被认为是单调递增的

    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            return False

    return True

def get_computer_time_diff(date, error_threshold):
    time_diff_record_root = '/share/hlyang/results/record'
    time_diff_data_path = os.path.join(time_diff_record_root, f'{date}_2m1.txt')

    assert os.path.exists(time_diff_data_path), time_diff_data_path
    with open(time_diff_data_path, 'r') as f:
        lines = f.readlines()
    time_diff_list = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            time_diff = parts[0]
            time_diff_list.append(int(time_diff))

    # TODO: 解决帧数相同，但两台电脑算出的时间戳之差相差太大
    assert len(time_diff_list) > 0

    # time_diff_mean = np.array(time_diff_list).mean().astype(np.int32)

    time_diff_array = np.array(time_diff_list)
    cnt_list = []
    for time_diff in time_diff_array:
        cnt = np.sum((time_diff_array >= time_diff - error_threshold) & (time_diff_array <= time_diff + error_threshold))
        cnt_list.append(cnt)
    cnt_array = np.array(cnt_list)
    max_idx = np.argmax(cnt_array)
    reasonable_time_diff = time_diff_array[max_idx]

    # for time_diff in time_diff_list:
    #     assert abs(time_diff - time_diff_mean) <= error_threshold

    return reasonable_time_diff

def cal_common_timestamps(timestamps_list, error_threshold=14):
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
            else: # 多个匹配项？
                print(camera_list[t_idx + 1], within_range)
                res = within_range[0]
                modified_cm_ts = (timestamp + res) // 2
                common_timestamps_.append(modified_cm_ts)
                # raise ValueError(f'len(within_range) should be 0 or 1, but got {len(within_range)}')

        common_timestamps = np.array(common_timestamps_)

    return common_timestamps.tolist()

def divide_timestamps_list(timestamps, divide_threshold = 35):
    '''
    将匹配出的长的公共时间戳序列，根据间隔切成多个不同视频所属的子公共时间戳序列。
    '''
    timestamps_list = []

    last_idx = 0
    for i in range(1, len(timestamps)):
        if abs(timestamps[i] - timestamps[i-1]) >= divide_threshold:
            timestamps_list.append(timestamps[last_idx:i])
            last_idx = i

    timestamps_list.append(timestamps[last_idx:])
    return timestamps_list

def process_frame_loss_global(camera_list, root_dir, video_list, time_diff, error_threshold):
    date = video_list[0][:8]

    assert os.path.exists(root_dir)
    for video_id in video_list:
        video_dir = os.path.join(root_dir, video_id, 'rgb')
        assert os.path.exists(video_dir), video_dir

    file_integrity_video_list = []
    # check if files exist
    for video_id in tqdm(video_list):
        integrity_bool = True
        video_dir = os.path.join(root_dir, video_id, 'rgb')

        for camera_id in camera_list:
            timestamp_path = os.path.join(video_dir, camera_id + '_FrameTimeStamp.txt')
            video_path =  os.path.join(video_dir, camera_id + '.mp4')
            if not os.path.exists(timestamp_path) or not os.path.exists(video_path):
                integrity_bool = False
                break

        if integrity_bool:
            file_integrity_video_list.append(video_id)
        else:
            pass
            # valid_record_path = os.path.join('/share/hlyang/results/record', f'{date}_invalid_video_id.txt')
            # with open(valid_record_path, 'a') as f:
            #     f.write(f'{video_id}\n')

    # timestamps_list = [[]] * len(camera_list)
    timestamps_list = [ [] for _ in range(len(camera_list))]

    for video_id in tqdm(file_integrity_video_list):
        video_dir = os.path.join(root_dir, video_id, 'rgb')
        for c_idx, camera_id in enumerate(camera_list):
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

            timestamps_list[c_idx] += timestamps

    computer1_camera_list = ['21218078', '22139906', '22139908', '22139910', '22139911', '22139913', '22139914', '22139946']
    computer2_camera_list = [camera for camera in camera_list if camera not in computer1_camera_list]
    assert len(computer2_camera_list) == 4
    computer1_ids = [camera_list.index(camera) for camera in computer1_camera_list]
    computer2_ids = [camera_list.index(camera) for camera in computer2_camera_list]

    # 使用计算出的误差
    modified_timestamps_list = []
    for idx, timestamps in enumerate(timestamps_list):
        if idx in computer2_ids:
            modified_timestamps = [timestamp - time_diff for timestamp in timestamps]
            modified_timestamps_list.append(modified_timestamps)
        else:
            modified_timestamps_list.append(timestamps)

    # 验证是不是单调递增，结果：不是！TODO:检查为什么不是
    # for timestamps in modified_timestamps_list:
    #     assert is_monotonic(timestamps)

    common_timestamps = cal_common_timestamps(modified_timestamps_list, error_threshold)

    common_timestamps_list = divide_timestamps_list(common_timestamps, divide_threshold=1000)
    for common_ts in common_timestamps_list:
        print(common_ts)
    print(len(common_timestamps_list))

if __name__ == '__main__':

    # camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']


    date = '20231010'
    root_dir = f'/share/datasets/HOI-mocap/{date}'

    dir_list = os.listdir(root_dir)
    video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]

    video_list.sort()
    print(video_list)

    error_threshold = 14

    timestamp_diff_2m1_mean = get_computer_time_diff(date, error_threshold)

    process_frame_loss_global(camera_list, root_dir, video_list, timestamp_diff_2m1_mean, error_threshold)

    # for video_id in tqdm(video_list):
    #     try:
    #         process_frame_loss_global(camera_list, root_dir, video_id, timestamp_diff_2m1_mean, error_threshold)
    #     except Exception as error:
    #         traceback.print_exc()
    #         print(error)
    #         continue

    # video_list = get_valid_video_list(date)
    # for video_id in tqdm(video_list):
    #     # cp calibration file
    #     src_cali_path = os.path.join(root_dir, video_id, 'src', 'calibration.json')
    #     os.path.exists(src_cali_path)
    #     dst_src_dir = os.path.join('/share/hlyang/results', video_id, 'src')
    #     os.makedirs(dst_src_dir, exist_ok=True)
    #     dst_src_path = os.path.join(dst_src_dir, 'calibration.json')

    #     time2 = time()