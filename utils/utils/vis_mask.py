'''
可视化标注结果，插值到(3000, 4096)大小再进行可视化。

example:
python utils/vis_mask.py --video_id 20230715_15
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
from hoi_io import load_bg_img

def vis_mask(camera_id, video_id, img_root, anno_results_dir, save_root, res_prefix = '', res_suffix = ''):
    
    assert os.path.exists(img_root)
    assert os.path.exists(anno_results_dir)
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, camera_id)
    os.makedirs(save_dir, exist_ok=True)
    
    camera_id
    anno_path = os.path.join(anno_results_dir, video_id+'|'+camera_id+'.npy')
    assert os.path.exists(anno_path)
    anno_res = np.load(anno_path)
    
    num_frame = anno_res.shape[0]
    for i in tqdm(range(num_frame)):
        frame_cnt = i + 1
        frame_id = str(frame_cnt).zfill(5)
        mask = anno_res[i]
        bg = load_bg_img(video_id, camera_id, frame_id)
        
        left_hand_mask = np.where(mask == 1, 1, 0).astype('uint8')
        interpolated_img = cv2.resize(left_hand_mask, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        bg[interpolated_img == 1, 0] = 255
        bg[interpolated_img == 1, 1:] = bg[interpolated_img == 1, 1:] / 2
        
        right_hand_mask = np.where(mask == 2, 1, 0).astype('uint8')
        interpolated_img = cv2.resize(right_hand_mask, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        bg[interpolated_img == 1, 1] = 255
        bg[interpolated_img == 1, 0] = bg[interpolated_img == 1, 0]/2
        bg[interpolated_img == 1, 2] = bg[interpolated_img == 1, 2]/2
        
        object1_mask = np.where(mask == 3, 1, 0).astype('uint8')
        interpolated_img = cv2.resize(object1_mask, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        bg[interpolated_img == 1, 2] = 255
        bg[interpolated_img == 1, 0:2] = bg[interpolated_img == 1, 0:2]/2
        
        object2_mask = np.where(mask == 4, 1, 0).astype('uint8')
        interpolated_img = cv2.resize(object2_mask, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        bg[interpolated_img == 1, 1:2] = 255
        bg[interpolated_img == 1, 0] = bg[interpolated_img == 1, 0]/2
        
        img_save_path = os.path.join(save_dir, res_prefix + frame_id + res_suffix + '.png')
        cv2.imwrite(img_save_path, bg)
    
if __name__ == "__main__":
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()
    video_id = args.video_id
    
    img_root = os.path.join('.','results', video_id, 'imgs')
    os.path.exists(img_root)
    
    mask_root = os.path.join('/share/hlyang/results', video_id, 'mask')
    os.makedirs(mask_root, exist_ok=True)
    
    anno_results_dir = os.path.join('.','results', video_id, 'original_anno_results')
    os.path.exists(anno_results_dir)
    
    procs = []
    
    for camera_id in camera_list:
        args = (camera_id, video_id, img_root, anno_results_dir, mask_root, camera_id+'_')
        proc = mlp.Process(target=vis_mask, args=args)
        proc.start()
        procs.append(proc)
        
    for i in range(len(procs)):
        procs[i].join()