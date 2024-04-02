import os
import sys
sys.path.append('.')
from utils.hoi_io import get_valid_video_list, get_num_frame
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

    # parser = argparse.ArgumentParser()
    date = '20230929'
    video_list = get_valid_video_list(date, consider_pipiline_failed=True)

    results_root = f'/share/hlyang/results/backup/{date}_backup'
    # dataset_root = '/share/hlyang/results/dataset'

    # dataset_date_dir = os.path.join(dataset_root, date)
    # os.makedirs(dataset_date_dir, exist_ok=True)
    
    pipeline_results_check_failed_record_path = f'/share/hlyang/results/record/{date}_results_check_failed.txt'

    for video_id in tqdm(video_list):
        try:
            # num_frame = get_num_frame(video_id)
            # start = 1
            # end = num_frame
            # frame_list = [str(frame).zfill(5) for frame in range(start, end + 1)]

            video_src_root = os.path.join(results_root, video_id)
            assert os.path.exists(video_src_root)
            
            metadata_dir = os.path.join(video_src_root, 'metadata')
            metadata_list = [filename for filename in os.listdir(metadata_dir) if filename.endswith('.pkl')]
            assert len(metadata_list) > 0
            metadata_path = os.path.join(metadata_dir, metadata_list[0])

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            num_frame = metadata['num_frame']

            # hand mesh
            right_hand_mesh_src_dir = os.path.join(video_src_root, 'get_world_mesh_from_mano_params', 'meshes', 'right_hand')
            left_hand_mesh_src_dir = os.path.join(video_src_root, 'get_world_mesh_from_mano_params', 'meshes', 'left_hand')

            right_hand_mesh_last_path = os.path.join(right_hand_mesh_src_dir, f'hand_{str(num_frame).zfill(5)}.obj')
            left_hand_mesh_last_path = os.path.join(left_hand_mesh_src_dir, f'hand_{str(num_frame).zfill(5)}.obj')
            assert os.path.exists(right_hand_mesh_last_path), right_hand_mesh_last_path
            assert os.path.exists(left_hand_mesh_last_path), left_hand_mesh_last_path

        except Exception as err:
            traceback.print_exc()
            print(err)
            add_a_line(pipeline_results_check_failed_record_path, f'{video_id}\n')
            continue
        
