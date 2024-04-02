import os
import sys
sys.path.append('.')
from utils.hoi_io2 import get_valid_video_list, get_num_frame, get_num_frame_v2
from utils.organize_dataset import add_a_line, organize_record_file

if __name__ == "__main__":
    
    # date_list = ['20230917', '20230919', '20230923', '20230926', '20230927', '20230928', '20230929', '20231002', '20231005', '20231006', '20231010', '20231013', '20231015']
    # date_list = ['20231019', '20231020', '20231026', '20231027', '20231031', '20231102']
    # date_list = ['20231103', '20231104', '20231105']
    date_list = ['20230930']
    # date_list = ['20231024']
    
    # video_record_root = '/share/hlyang/results'
    video_record_root = '/data2/hlyang/results'
    record_root = '/data2/hlyang/results'
    upload_root = '/data2/HOI-mocap'
    
    for date in date_list:
        video_list = get_valid_video_list(video_record_root, date, remove_hand=True)
        nokov_success_record_path = os.path.join(record_root, 'record', f'{date}_nokov_succeed.txt')
        
        for video_id in video_list:
            obj_pose_dir = os.path.join(upload_root, 'HO_poses', date, video_id, 'objpose')
            invalid_path = os.path.join(upload_root, 'HO_poses', date, video_id, 'invalid')
            if os.path.isdir(obj_pose_dir) and not os.path.isfile(invalid_path):
                add_a_line(nokov_success_record_path, f'{video_id}')
                
        if os.path.exists(nokov_success_record_path):
            organize_record_file(nokov_success_record_path)