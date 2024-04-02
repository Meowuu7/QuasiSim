'''
可视化标注结果，可视化在downsample后的域大小。

example:
python utils/vis_mask2.py --video_id 20230715_15
'''

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import cv2
from tqdm import tqdm
import argparse
import pickle
import multiprocessing as mlp
import numpy as np
from hoi_io import load_bg_img, get_downsampled_seg_infos_batch
from scandir import scandir
    
def vis_mask(camera_id, video_id, anno_results_dir, save_root, res_prefix = '', res_suffix = ''):
    assert os.path.exists(anno_results_dir)
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, camera_id)
    os.makedirs(save_dir, exist_ok=True)
    
    metadata_path = os.path.join('/share/hlyang/results', video_id, 'metadata', camera_id + '.pkl')
    assert os.path.exists(metadata_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    num_frame = metadata['num_frame']
    
    frame_list = [str(i+1).zfill(5) for i in range(num_frame)]
    seg, downsample_factor = get_downsampled_seg_infos_batch(video_id, frame_list, [camera_id])
    downsampled_width = 4096 // downsample_factor
    downsample_height = 3000 // downsample_factor
    # seg_right_hand = np.where(seg == 1, 1, 0)
    # seg_left_hand = np.where(seg == 2, 1, 0)
    # seg_object1 = np.where(seg == 3, 1, 0)
    # seg_object2 = np.where(seg == 4, 1, 0)
    
    num_frame = seg.shape[0]
    for i in tqdm(range(num_frame)):
        frame_cnt = i + 1
        frame_id = str(frame_cnt).zfill(5)
        mask = seg[i, 0]
        bg = load_bg_img(video_id, camera_id, frame_id)
        bg = cv2.resize(bg, (downsampled_width, downsample_height), interpolation=cv2.INTER_LINEAR)
        
        right_hand_mask = np.where(mask == 1, 1, 0).astype('uint8')
        bg[right_hand_mask == 1, 2] = 255
        bg[right_hand_mask == 1, 0] = bg[right_hand_mask == 1, 0]/2
        bg[right_hand_mask == 1, 1] = bg[right_hand_mask == 1, 1]/2
        
        left_hand_mask = np.where(mask == 2, 1, 0).astype('uint8')
        bg[left_hand_mask == 1, 0] = 255
        bg[left_hand_mask == 1, 1:] = bg[left_hand_mask == 1, 1:] / 2
        
        object1_mask = np.where(mask == 3, 1, 0).astype('uint8')
        bg[object1_mask == 1, 0] = 255
        bg[object1_mask == 1, 2] = 255
        bg[object1_mask == 1, 1] = bg[object1_mask == 1, 1]/2
        
        object2_mask = np.where(mask == 4, 1, 0).astype('uint8')
        bg[object2_mask == 1, 1] = 255
        bg[object2_mask == 1, 0] = bg[object2_mask == 1, 0]/2
        bg[object2_mask == 1, 2] = bg[object2_mask == 1, 2]/2
        
        img_save_path = os.path.join(save_dir, res_prefix + frame_id + res_suffix + '.png')
        cv2.imwrite(img_save_path, bg)
    
if __name__ == "__main__":
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()
    video_id = args.video_id
    
    mask_root = os.path.join('/share/hlyang/results', video_id, 'mask')
    os.makedirs(mask_root, exist_ok=True)
    
    anno_results_dir = os.path.join('.','results', video_id, 'anno_results')
    os.path.exists(anno_results_dir)
    
    procs = []
    
    for camera_id in camera_list:
        args = (camera_id, video_id, anno_results_dir, mask_root, camera_id+'_')
        proc = mlp.Process(target=vis_mask, args=args)
        proc.start()
        procs.append(proc)
        
    for i in range(len(procs)):
        procs[i].join()