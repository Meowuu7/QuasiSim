import traceback
import sys
sys.path.append('.')
import os
import shutil
from tqdm import tqdm

def copy_file_or_dir(src_path, dst_path):
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)
    else:
        shutil.copytree(src_path, dst_path)
    print(f'{src_path} -> {dst_path}')

def load_sequence_names_from_organized_record(path: str, date: str):
    organized_sequence_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0 and parts[0].startswith(date):
            organized_sequence_list.append(parts[0])
            
    organized_sequence_list = list(set(organized_sequence_list))
    organized_sequence_list.sort(key=lambda x:int(x))
    
    return organized_sequence_list

def get_organized_date_list(path: str):
    organized_date_list = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            organized_date_list.append(parts[0][:8])
    
    organized_date_list = list(set(organized_date_list))
    organized_date_list.sort()
    
    return organized_date_list

if __name__ == '__main__':
    
    # dataset_src_root = '/share/hlyang/results/dataset'
    # dataset_dst_root = '/data2/hlyang/results/dataset'
    
    # organized_dataset_record_path = '/share/hlyang/results/dataset/organized_record.txt'
    
    dataset_src_root = '/data2/hlyang/results/dataset'
    dataset_dst_root = '/data3/hlyang/results/dataset'
    
    organized_dataset_record_path = '/data2/hlyang/results/dataset/organized_record.txt'
    
    date_list = get_organized_date_list(organized_dataset_record_path)
    for date in date_list:
        video_list = load_sequence_names_from_organized_record(organized_dataset_record_path, date)
        for video_id in tqdm(video_list):
            sequence_src_dir = os.path.join(dataset_src_root, date, video_id)
            sequence_dst_dir = os.path.join(dataset_dst_root, date, video_id)
            os.makedirs(sequence_dst_dir, exist_ok=True)
            
            dir_list = ['interaction_field', 'mano_wo_contact', 'rgb', 'src', 'egocentric_rgb.mp4']
            for dir in dir_list:
                src_dir = os.path.join(sequence_src_dir, dir)
                dst_dir = os.path.join(sequence_dst_dir, dir)
                
                # if os.path.exists(dst_dir):
                    # continue
                # shutil.copytree(src_dir, dst_dir)
                # print(f'{src_dir} -> {dst_dir}')
                copy_file_or_dir(src_dir, dst_dir)