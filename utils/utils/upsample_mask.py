'''
由于标注工具有downsample，先把标注结果upsample一下。

example:
python utils/upsample_mask.py --video_id 20230715_15
'''

import numpy as np
import cv2
import argparse
import os
import multiprocessing as mlp
from tqdm import tqdm

def upsample_mask(video_id, camera_id):
    path = os.path.join('/share/hlyang/results', video_id, 'original_anno_results', video_id+'|'+camera_id+'.npy')
    mask = np.load(path) # [num_frame, ]
    upsampled_mask_list = []
    for idx in tqdm(range(mask.shape[0])):
        # 0和4能插出2来，肯定不行
        # upsampled_mask_ = cv2.resize(mask[idx, ...], (4096, 3000), interpolation=cv2.INTER_LINEAR)
        left_hand = np.where(mask[idx, ...] == 1, 1, 0).astype(np.uint8)
        left_hand = cv2.resize(left_hand, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        right_hand = np.where(mask[idx, ...] == 2, 1, 0).astype(np.uint8)
        right_hand = cv2.resize(right_hand, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        object1 = np.where(mask[idx, ...] == 3, 1, 0).astype(np.uint8)
        object1 = cv2.resize(object1, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        object2 = np.where(mask[idx, ...] == 4, 1, 0).astype(np.uint8)
        object2 = cv2.resize(object2, (4096, 3000), interpolation=cv2.INTER_LINEAR)
        upsampled_mask_ = np.zeros((3000, 4096), dtype=np.uint8)
        upsampled_mask_[left_hand == 1] = 1
        upsampled_mask_[right_hand == 1] = 2
        upsampled_mask_[object1 == 1] = 3
        upsampled_mask_[object2 == 1] = 4
        
        upsampled_mask_list.append(upsampled_mask_)
    upsampled_mask = np.stack(upsampled_mask_list, axis=0)
    print(upsampled_mask.shape)
    save_path = os.path.join('/share/hlyang/results', video_id, 'anno_results', video_id+'|'+camera_id+'.npy')
    np.save(save_path, upsampled_mask)

if __name__ == '__main__':
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()
    video_id = args.video_id
    
    os.makedirs(os.path.join('/share/hlyang/results', video_id, 'anno_results'), exist_ok=True)
    
    procs = []
    for camera_id in camera_list:
        args = (video_id, camera_id)
        proc = mlp.Process(target=upsample_mask, args=args)
        proc.start()
        procs.append(proc)
        
    for i in range(len(procs)):
        procs[i].join()
    
    # data = np.load('./20230715_15|22070938.npy')
    # data = data[0]
    # img = np.zeros((750,1024,3))
    
    # data = np.where(data == 1, 1, 0).astype('uint8')
    # interpolated_img = cv2.resize(data, (4096, 3000), interpolation=cv2.INTER_LINEAR)
    # print(interpolated_img.shape)
    
    # color_img = np.zeros((3000, 4096, 3), dtype=np.uint8)
    # color_img[interpolated_img == 1] = [0, 0, 255]
    
    # cv2.imwrite('./test.jpg', color_img)
