'''
只crop可扣除一只手的图，没有resize

example:
python utils/test_seg.py
'''

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import pickle
import multiprocessing as mlp
from hoi_io import get_seg_infos_batch3, load_bg_img
from scandir import scandir
import argparse

if __name__ == '__main__':
    video_id = '20230715_15'
    frame_list = [str(i+1).zfill(5) for i in range(1)]
    camera_list = ['22070938']
    seg = get_seg_infos_batch3(video_id, frame_list, camera_list)
    
    left_hand_seg = np.where(seg[0,0] == 1, 1, 0)
    right_hand_seg = np.where(seg[0,0] == 2, 1, 0)
    
    left = np.zeros((3000, 4096, 3), dtype=np.uint8)
    right = np.zeros((3000, 4096, 3), dtype=np.uint8)
    
    left[left_hand_seg == 1] = [255,0,0]
    right[right_hand_seg == 1] = [255,0,0]
    
    cv2.imwrite('./left.png', left)
    cv2.imwrite('./right.png', right)

    # seg = np.load('/share/hlyang/results/20230715_15/original_anno_results/20230715_15|22070938.npy')
    # left_hand_seg = np.where(seg[0] == 1, 1, 0)
    # right_hand_seg = np.where(seg[0] == 2, 1, 0)
    
    # left = np.zeros((750, 1024, 3), dtype=np.uint8)
    # right = np.zeros((750, 1024, 3), dtype=np.uint8)
    
    # left[left_hand_seg == 1] = [255,0,0]
    # right[right_hand_seg == 1] = [255,0,0]
    
    # cv2.imwrite('./left.png', left)
    # cv2.imwrite('./right.png', right)