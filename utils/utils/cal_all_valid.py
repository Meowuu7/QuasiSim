import os
import sys
sys.path.append('.')
from utils.hoi_io import get_valid_video_list, get_num_frame, get_num_frame_v2
from utils.get_world_mesh_from_mano_params import load_world_meshes_acc
import argparse
import shutil
from tqdm import tqdm
import pickle
import traceback

def add_a_line(path, line):
    with open(path, 'a') as f:
        f.write(line)

if __name__ == "__main__":

    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']
    root = '/share/hlyang/results'

    # parser = argparse.ArgumentParser()
    # date = '20230917'
    date_list = ['20230917', '20230919', '20230923', '20230926', '20230927', '20230928', '20230929', '20231002', '20231005', '20231006', '20231010', '20231013', '20231015']
    cnt = 0
    
    for date in date_list:
        video_list = get_valid_video_list(date, remove_hand=True)

        for video_id in tqdm(video_list):
            try:
                num_frame = get_num_frame_v2(video_id)
                start = 1
                end = num_frame
                # frame_list = [str(frame).zfill(5) for frame in range(start, end + 1)]
                last_frame = str(end).zfill(5)

                video_src_root = os.path.join(root, date, video_id)
                os.path.exists(video_src_root)

                # hand mesh
                mesh_exp_name = 'world_mesh_batch'
                right_hand_mesh = load_world_meshes_acc(date, video_id, mesh_exp_name, [last_frame], right_hand_bool=True)[last_frame]
                left_hand_mesh = load_world_meshes_acc(date, video_id, mesh_exp_name, [last_frame], right_hand_bool=False)[last_frame]
                assert right_hand_mesh is not None
                assert left_hand_mesh is not None
                
                cnt += 1

            except Exception as err:
                # traceback.print_exc()
                # print(err)
                continue
    
    print(cnt)