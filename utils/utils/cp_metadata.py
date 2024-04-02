import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    date_list = ['20230923', '20230926', '20230927', '20230928', '20230929', '20231002']
    
    for date in date_list:
        
        src_root = os.path.join('/share/hlyang/results', f'{date}_backup')
        dst_root = os.path.join('/share/hlyang/results', date)
        os.makedirs(dst_root, exist_ok=True)
        
        video_id_list = os.listdir(src_root)
        video_id_list.sort()
        
        dir_list = ['metadata', 'src']
        
        for video_id in tqdm(video_id_list):
            src_video_root = os.path.join(src_root, video_id)
            dst_video_root = os.path.join(dst_root, video_id)
            os.makedirs(dst_video_root, exist_ok=True)
            
            for dir in dir_list:
                src_dir = os.path.join(src_video_root, dir)
                dst_dir = os.path.join(dst_video_root, dir)
                shutil.copytree(src_dir, dst_dir)