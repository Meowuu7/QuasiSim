import os
import sys
sys.path.append('.')
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    filename_list = ['right_hand_shape.pkl', 'left_hand_shape.pkl']
    
    root_src = '/share/hlyang/results'
    # root_src = '/data2/hlyang/results'
    src_date = '20231010'
    src_dir = f'{root_src}/{src_date}/{src_date}_hand2/src'
    
    root_dst = '/data2/hlyang/results'
    dst_date = '20230930'
    dst_dir_list = [f'{root_dst}/{dst_date}/{dst_date}_{str(i).zfill(3)}/src' for i in range(121, 153)]
    
    for dst_dir in tqdm(dst_dir_list):
        if os.path.isdir(dst_dir):
            for filename in filename_list:
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)
                shutil.copy(src_path, dst_path)