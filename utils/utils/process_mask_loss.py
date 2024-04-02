'''
有一些帧缺少了某一种mask，所以在metadata中加上key：frame_without_left_hand, frame_without_right_hand, frame_without_object1, frame_without_object2。以后的一些操作会跳过这里面的帧。

TODO：预计被跳过的帧全部使用前一帧的结果做平滑处理。

example: 
python utils/process_mask_loss.py --video_id 20230715_15
'''
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import argparse
import pickle
import numpy as np
from hoi_io import get_downsampled_seg_infos_batch


if __name__ == '__main__':
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()
    video_id = args.video_id
    
    video_meta_dir = os.path.join('/share/hlyang/results', video_id, 'metadata')
    assert os.path.exists(video_meta_dir)
    
    for camera in camera_list:
        metadata_path = os.path.join(video_meta_dir, f'{camera}.pkl')
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        num_frame = data['num_frame']
        frame_list = [str(i).zfill(5) for i in range(1, num_frame + 1)]
        
        seg, downsample_factor = get_downsampled_seg_infos_batch(video_id, frame_list, [camera])
        right_hand_seg = np.where(seg == 1, 1, 0).astype(np.uint8)
        left_hand_seg = np.where(seg == 2, 1, 0).astype(np.uint8)
        object1_seg = np.where(seg == 3, 1, 0).astype(np.uint8)
        object2_seg = np.where(seg == 4, 1, 0).astype(np.uint8)
        
        frame_without_right_hand = []
        frame_without_left_hand = []
        frame_without_object1 = []
        frame_without_object2 = []
        for i, frame_id in enumerate(frame_list):
            if np.count_nonzero(right_hand_seg[i, 0, ...]) == 0:
                frame_without_right_hand.append(frame_id)
            if np.count_nonzero(left_hand_seg[i, 0, ...]) == 0:
                frame_without_left_hand.append(frame_id)
            if np.count_nonzero(object1_seg[i, 0, ...]) == 0:
                frame_without_object1.append(frame_id)
            if np.count_nonzero(object2_seg[i, 0, ...]) == 0:
                frame_without_object2.append(frame_id)
        print(camera)
        print(frame_without_right_hand)
        print(frame_without_left_hand)
        print(frame_without_object1)
        print(frame_without_object2)
        data['frame_without_right_hand'] = frame_without_right_hand
        data['frame_without_left_hand'] = frame_without_left_hand
        data['frame_without_object1'] = frame_without_object1
        data['frame_without_object2'] = frame_without_object2

        with open(metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        
    
        
            
    
        
    