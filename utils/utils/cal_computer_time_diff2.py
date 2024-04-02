'''
解析录制时的{camera_id}__FrameTimeStamp.txt，根据timestamp丢弃录制时有丢失的帧，并将剩余的帧数据存为pkl。之后video2img.py等一系列程序将基于该pkl进行处理。

生成{date}_record.txt {date}_2m1.txt

TODO：不同的相机从第一帧开始timestamp就不同，应该考虑这个误差。
TODO：完善并实装该功能。目前仅仅统计每一个视角的frame数量。

example:
python utils/process_frame_loss.py --root_dir /share/datasets/HOI-mocap/20230904 --video_id 20230904_01
'''

import traceback
import argparse
import os
import pickle
from shutil import copy
from tqdm import tqdm
import numpy as np

time_diff = None

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
            else: # ？？？
                res = within_range[np.abs(within_range - timestamp).argmin()]
                modified_cm_ts = (timestamp + res) // 2
                common_timestamps_.append(modified_cm_ts)
                # raise ValueError(f'len(within_range) should be 0 or 1, but got {len(within_range)}')
            
        common_timestamps = np.array(common_timestamps_)

    return common_timestamps.tolist()

def cal_computer_time_diff(camera_list, date_root_dir, save_root, video_id, error_threshold):
    date = video_id[:8]
    global time_diff

    time_diff_record_root = os.path.join(save_root, 'record')
    time_diff_reference_record_root = os.path.join(save_root, 'reference_record')
    # time_diff_record_root = '/share/hlyang/results/record'
    # time_diff_reference_record_root = '/share/hlyang/results/reference_record'
    time_diff_record_path = os.path.join(time_diff_reference_record_root, f'{date}_record.txt')
    time_diff_data_path = os.path.join(time_diff_record_root, f'{date}_2m1.txt')

    assert os.path.exists(date_root_dir)
    video_dir = os.path.join(date_root_dir, video_id, 'rgb')
    assert os.path.exists(video_dir), video_dir

    # check if files exist
    # for camera_id in camera_list:
    #     assert os.path.exists(os.path.join((video_dir, camera_id + '_FrameTimeStamp.txt'))
    #     assert os.path.exists(os.path.join((video_dir, camera_id + '.mp4'))

    computer1_camera_list = ['21218078', '22139906', '22139908', '22139910', '22139911', '22139913', '22139914', '22139946']
    computer2_camera_list = [camera for camera in camera_list if camera not in computer1_camera_list]
    assert len(computer2_camera_list) == 4
    computer1_ids = [camera_list.index(camera) for camera in computer1_camera_list]
    computer2_ids = [camera_list.index(camera) for camera in computer2_camera_list]

    timestamps_list = []
    for camera_id in camera_list:
        timestamp_path = os.path.join(video_dir, camera_id + '_FrameTimeStamp.txt')

        if not os.path.exists(timestamp_path):
            record_line = f'{video_id} lack timestamp file\n'
            with open(time_diff_record_path, 'a') as f:
                f.write(record_line)
            return

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

    computer1_timestamps_list = [timestamps_list[idx] for idx in computer1_ids]
    computer2_timestamps_list = [timestamps_list[idx] for idx in computer2_ids]
    computer1_common_timestamps = cal_common_timestamps(computer1_timestamps_list, error_threshold)
    computer2_common_timestamps = cal_common_timestamps(computer2_timestamps_list, error_threshold)

    len_computer1_common_timestamps = len(computer1_common_timestamps)
    len_computer2_common_timestamps = len(computer2_common_timestamps)
    # print(computer1_common_timestamps[0], computer2_common_timestamps[0])

    record_line = f'{video_id} '
    for idx, camera in enumerate(camera_list):
        record_line += f'{camera}: {len(timestamps_list[idx])} '
    record_line += f'computer1: {len_computer1_common_timestamps} computer2: {len_computer2_common_timestamps}\n'

    with open(time_diff_record_path, 'a') as f:
        f.write(record_line)

    if len_computer1_common_timestamps == len_computer2_common_timestamps:
        time_diff_2m1 = computer2_common_timestamps[0] - computer1_common_timestamps[0]

        with open(time_diff_data_path, 'a') as f:
            f.write(f'{time_diff_2m1} {video_id}\n')

        time_diff = time_diff_2m1

        if time_diff is not None:
            assert abs(time_diff_2m1 - time_diff) <= error_threshold


    return
    # if len_computer1_common_timestamps != len_computer2_common_timestamps:
        # return
        
if __name__ == '__main__':

    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']

    date = '20230930'
    # upload_root_dir = '/share/datasets/HOI-mocap'
    upload_root_dir = '/data2/HOI-mocap'
    upload_date_root = os.path.join(upload_root_dir, date)
    
    # save_root = '/share/hlyang/results'
    save_root = '/data2/hlyang/results'

    dir_list = os.listdir(upload_date_root)
    video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]
    video_list.sort()
    
    print(video_list)

    error_threshold = 15

    for video_id in tqdm(video_list):
        try:
            cal_computer_time_diff(camera_list, upload_date_root, save_root, video_id, error_threshold)
        except Exception as error:
            traceback.print_exc()
            continue
