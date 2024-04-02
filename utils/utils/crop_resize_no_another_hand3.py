'''
只crop可扣除一只手的图，没有resize。

example:
python utils/crop_resize_no_another_hand.py --video_id 20230818_04_old

TODO：检查为什么运行到后面会越来越慢。似乎是卡在了load_bg_img，可以测一下这一行的时间。 re:好像只是单纯有点慢
'''

import os
import sys
sys.path.append('.')
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import pickle
import multiprocessing as mlp
from utils.hoi_io2 import get_downsampled_seg_infos_batch, get_seg_infos_batch3, load_bg_img, get_downsampled_seg_infos_batch_v2, read_init_crop, cal_represent_frame_list, get_downsampled_seg_infos_batch_v2_acc_batch
from utils.scandir import scandir
import argparse
from time import time
import json

def crop_from_mask(video_id: str, camera_list: str, frame_list: list[str]):
    '''
    当前帧如果没有某个mask，则用上一帧的bbox进行crop。
    '''

    seg, downsample_factor = get_downsampled_seg_infos_batch(video_id, frame_list, camera_list)
    right_hand_seg = np.where(seg == 1, 1, 0).astype(np.uint8)
    left_hand_seg = np.where(seg == 2, 1, 0).astype(np.uint8)

    last_right_min_x = None
    last_right_max_x = None
    last_right_min_y = None
    last_right_max_y = None
    last_right_meanh = None
    last_right_meanw = None

    last_left_min_x = None
    last_left_max_x = None
    last_left_min_y = None
    last_left_max_y = None
    last_left_meanh = None
    last_left_meanw = None

    MAX_HEIGHT = 4095
    MAX_WIDTH = 2999
    MARGIN_SIZE = 50

    # TODO：以下的x和y定义反了，需要更改
    for camera_idx, camera_id in enumerate(camera_list):
        # left_hand_crop_dir = osp.join(root, video_id, 'crop_imgs_left_hand', camera_id)
        # right_hand_crop_dir = osp.join(root, video_id, 'crop_imgs_right_hand', camera_id)
        left_hand_crop_dir = osp.join(root, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = osp.join(root, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)


        for frame_idx, frame_id in tqdm(enumerate(frame_list)):
            mask_right_hand = right_hand_seg[frame_idx, camera_idx]
            mask_left_hand = left_hand_seg[frame_idx, camera_idx]

            # upsample
            mask_right_hand = cv2.resize(mask_right_hand, (4096, 3000), interpolation=cv2.INTER_LINEAR)
            mask_left_hand = cv2.resize(mask_left_hand, (4096, 3000), interpolation=cv2.INTER_LINEAR)

            # 读入图片
            # timer1 = time()
            img = load_bg_img(video_id, camera_id, frame_id)
            # timer2 = time()
            # print(timer2 - timer1)

            # 以下为右手
            mask_right_hand_idx = np.nonzero(mask_right_hand)
            if len(mask_right_hand_idx[0]) != 0:
                right_minh = np.min(mask_right_hand_idx[0])
                right_maxh = np.max(mask_right_hand_idx[0])
                right_minw = np.min(mask_right_hand_idx[1])
                right_maxw = np.max(mask_right_hand_idx[1])
                right_midh = (right_maxh + right_minh) // 2
                right_midw = (right_maxw + right_minw) // 2
                right_meanh = np.mean(mask_right_hand_idx[0])
                right_meanw = np.mean(mask_right_hand_idx[1])

                right_length = max(right_maxh - right_minh, right_maxw - right_minw)
                right_min_x = max(0, right_midh - right_length // 2 - MARGIN_SIZE)
                right_max_x = min(MAX_WIDTH, right_midh + right_length // 2 + MARGIN_SIZE)
                right_min_y = max(0, right_midw - right_length // 2 - MARGIN_SIZE)
                right_max_y = min(MAX_HEIGHT, right_midw + right_length // 2 + MARGIN_SIZE)

                last_right_min_x = right_min_x
                last_right_max_x = right_max_x
                last_right_min_y = right_min_y
                last_right_max_y = right_max_y
                last_right_meanh = right_meanh
                last_right_meanw = right_meanw
            elif frame_idx != 0 and len(mask_right_hand_idx[0]) == 0:
                right_min_x = last_right_min_x
                right_max_x = last_right_max_x
                right_min_y = last_right_min_y
                right_max_y = last_right_max_y
                right_meanh = last_right_meanh
                right_meanw = last_right_meanw
            # else:
            #     print('no hand mask in the first frame!')
            #     exit(1)
            else:
                print('no hand mask in the first frame of this sub_frame_list!')
                print('right_hand', camera_id, frame_id)
                if frame_idx == 0 and frame_id != '00001':
                    info_path = osp.join(right_hand_crop_dir, camera_id + '_' + str(int(frame_id)-1).zfill(5) + '_crop_info.pkl')
                    assert os.path.exists(info_path)
                    with open(info_path, 'rb') as f:
                        right_min_x, right_max_x, right_min_y, right_max_y, right_meanh, right_meanw = pickle.load(f)

                    last_right_min_x = right_min_x
                    last_right_max_x = right_max_x
                    last_right_min_y = right_min_y
                    last_right_max_y = right_max_y
                    last_right_meanh = right_meanh
                    last_right_meanw = right_meanw
                else:
                    print('GG, no hand mask in the first frame!')
                    exit(1)

            # 将左手变成白色
            right_crop = img.copy()
            right_crop[mask_left_hand == 1] = 255.0

            # 居中
            right_crop = right_crop[right_min_x: right_max_x, right_min_y: right_max_y]
            right_crop_info = [right_min_x, right_max_x, right_min_y, right_max_y, right_meanh, right_meanw]

            with open(osp.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                pickle.dump(right_crop_info, f)

            cv2.imwrite(osp.join(right_hand_crop_dir, camera_id + '_' +frame_id + '.png'), right_crop)

            # 以下为左手
            mask_left_hand_idx = np.nonzero(mask_left_hand)

            # test_img = np.zeros((3000, 4096, 3))
            # for y, x in zip(mask_left_hand_idx[0], mask_left_hand_idx[1]):
            #     cv2.circle(test_img, (x, y), 5, (255, 0, 0), -1)
            # cv2.imwrite('./test.png',test_img)

            if len(mask_left_hand_idx[0]) != 0:
                left_minh = np.min(mask_left_hand_idx[0])
                left_maxh = np.max(mask_left_hand_idx[0])
                left_minw = np.min(mask_left_hand_idx[1])
                left_maxw = np.max(mask_left_hand_idx[1])
                left_midh = (left_maxh + left_minh) // 2
                left_midw = (left_maxw + left_minw) // 2
                left_meanh = np.mean(mask_left_hand_idx[0])
                left_meanw = np.mean(mask_left_hand_idx[1])

                left_length = max(left_maxh - left_minh, left_maxw - left_minw)
                left_min_x = max(0, left_midh - left_length // 2 - MARGIN_SIZE)
                left_max_x = min(MAX_WIDTH, left_midh + left_length // 2 + MARGIN_SIZE)
                left_min_y = max(0, left_midw - left_length // 2 - MARGIN_SIZE)
                left_max_y = min(MAX_HEIGHT, left_midw + left_length // 2 + MARGIN_SIZE)

                last_left_min_x = left_min_x
                last_left_max_x = left_max_x
                last_left_min_y = left_min_y
                last_left_max_y = left_max_y
                last_left_meanh = left_meanh
                last_left_meanw = left_meanw
            elif frame_idx != 0 and len(mask_left_hand_idx[0]) == 0:
                left_min_x = last_left_min_x
                left_max_x = last_left_max_x
                left_min_y = last_left_min_y
                left_max_y = last_left_max_y
                left_meanh = last_left_meanh
                left_meanw = last_left_meanw
            else:
                print('no hand mask in the first frame of this sub_frame_list!')
                print('left_hand', camera_id, frame_id)
                if frame_idx == 0 and frame_id != '00001':
                    info_path = osp.join(left_hand_crop_dir, camera_id + '_' + str(int(frame_id)-1).zfill(5) + '_crop_info.pkl')
                    assert os.path.exists(info_path)
                    with open(info_path, 'rb') as f:
                        left_min_x, left_max_x, left_min_y, left_max_y, left_meanh, left_meanw = pickle.load(f)

                    last_left_min_x = left_min_x
                    last_left_max_x = left_max_x
                    last_left_min_y = left_min_y
                    last_left_max_y = left_max_y
                    last_left_meanh = left_meanh
                    last_left_meanw = left_meanw
                else:
                    print(frame_idx, frame_id)
                    print('GG, no hand mask in the first frame!')
                    exit(1)

            # 将右手变成白色
            left_crop = img.copy()
            left_crop[mask_right_hand == 1] = 255.0

            # 居中
            left_crop = left_crop[left_min_x: left_max_x, left_min_y: left_max_y]
            left_crop_info = [left_min_x, left_max_x, left_min_y, left_max_y, left_meanh, left_meanw]

            with open(osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                pickle.dump(left_crop_info, f)

            cv2.imwrite(osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png'), left_crop)

def crop(img, min_h, max_h, min_w, max_w, black_min_h=None, black_max_h=None, black_min_w=None, black_max_w=None):
    img = img.copy()

    mid_h = (min_h + max_h) // 2
    mid_w = (min_w + max_w) // 2
    length = max(max_h - min_h, max_w - min_w)

    if [black_min_h, black_max_h, black_min_w, black_max_w] == [not None, not None, not None, not None]:
        img[black_min_h: black_max_h, black_min_w, black_max_w] = 255.0

    crop = img[min_h: max_h, min_w: max_w]

    return crop

def apply_factor_and_margin(min_h, max_h, min_w, max_w, factor, margin_size, MAX_H, MAX_W):
    # # avoid overflow
    # min_h_ = min_h.astype(np.uint16)
    # max_h_ = max_h.astype(np.uint16)
    # min_w_ = min_w.astype(np.uint16)
    # max_w_ = max_w.astype(np.uint16)
    # mid_h = ((min_h_ + max_h_) // 2).astype(np.uint8)
    # mid_w = ((min_w_ + max_w_) // 2).astype(np.uint8)

    mid_h = (min_h + max_h) // 2
    mid_w = (min_w + max_w) // 2

    length = max(max_h - min_h, max_w - min_w)
    length = int(length * factor)
    half_length = length // 2

    min_h = max(0, mid_h - half_length - margin_size, 0)
    max_h = min(MAX_H, mid_h + half_length + margin_size)
    min_w = max(0, mid_w - half_length - margin_size, 0)
    max_w = min(MAX_W, mid_w + half_length + margin_size)

    return min_h, max_h, min_w, max_w, mid_h, mid_w

def crop_hand_from_pos(root, video_id: str, camera_id: str, frame_id: str, pos, crop_factor = 1, margin_size = 0):
    H = 3000
    W = 4096
    MAX_HEIGHT = H - 1
    MAX_WIDTH = W - 1

    assert pos.shape == (2, 4)
    # assert np.all(pos[...] >= 0), print(pos)
    # assert np.all(pos[..., [0, 1]] <= MAX_HEIGHT), print(pos)
    # assert np.all(pos[..., [2, 3]] <= MAX_WIDTH), print(pos)

    if np.all(pos[...] >= 0) and np.all(pos[..., [0, 1]] <= MAX_HEIGHT) and np.all(pos[..., [2, 3]] <= MAX_WIDTH):

        left_hand_crop_dir = osp.join(root, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = osp.join(root, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)

        bg = load_bg_img(video_id, camera_id, frame_id)

        # crop info
        right_crop_info_path = osp.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl')
        (right_min_h, right_max_h, right_min_w, right_max_w) = pos[0]
        right_min_h, right_max_h, right_min_w, right_max_w, right_mid_h, right_mid_w = apply_factor_and_margin(right_min_h, right_max_h, right_min_w, right_max_w, crop_factor, margin_size, MAX_HEIGHT, MAX_WIDTH)

        right_crop_info = [right_min_h, right_max_h, right_min_w, right_max_w, right_mid_h, right_mid_w]

        with open(right_crop_info_path, 'wb') as f:
            pickle.dump(right_crop_info, f)

        left_crop_info_path = osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl')
        (left_min_h, left_max_h, left_min_w, left_max_w) = pos[1]
        left_min_h, left_max_h, left_min_w, left_max_w, left_mid_h, left_mid_w = apply_factor_and_margin(left_min_h, left_max_h, left_min_w, left_max_w, crop_factor, margin_size, MAX_HEIGHT, MAX_WIDTH)

        left_crop_info = [left_min_h, left_max_h, left_min_w, left_max_w, left_mid_h, left_mid_w]
        with open(left_crop_info_path, 'wb') as f:
            pickle.dump(left_crop_info, f)

        right_crop = crop(bg, right_min_h, right_max_h, right_min_w, right_max_w, left_min_h, left_max_h, left_min_w, left_max_w)
        right_crop_path = osp.join(right_hand_crop_dir, camera_id + '_' + frame_id + '.png')
        cv2.imwrite(right_crop_path, right_crop)

        # left hand
        left_crop = crop(bg, left_min_h, left_max_h, left_min_w, left_max_w, right_min_h, right_max_h, right_min_w, right_max_w)
        left_crop_path = osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png')
        cv2.imwrite(left_crop_path, left_crop)

def crop_hand_from_pos_acc(root, local_root, date: str, crop_save_exp_name, crop_info_save_exp_name, video_id: str, camera_list: str, frame_list: str, represent_frame_id, img_batch, pos_batch, crop_factor = 1, margin_size = 0):
    H = 3000
    W = 4096
    MAX_HEIGHT = H - 1
    MAX_WIDTH = W - 1

    # assert pos.shape == (2, 4)
    # assert np.all(pos[...] >= 0), print(pos)
    # assert np.all(pos[..., [0, 1]] <= MAX_HEIGHT), print(pos)
    # assert np.all(pos[..., [2, 3]] <= MAX_WIDTH), print(pos)

    # TODO 
    right_crop_info_batch = {}
    left_crop_info_batch = {}
    
    for f_idx, frame_id in enumerate(frame_list):
        right_crop_info_batch[frame_id] = {}
        left_crop_info_batch[frame_id] = {}
        
        for c_idx, camera_id in enumerate(camera_list):
            if camera_id not in pos_batch[frame_id].keys():
                continue
            pos = pos_batch[frame_id][camera_id]
    
            if np.all(pos[...] >= 0) and np.all(pos[..., [0, 1]] <= MAX_HEIGHT) and np.all(pos[..., [2, 3]] <= MAX_WIDTH):

                left_hand_crop_dir = osp.join(local_root, date, video_id, crop_save_exp_name, 'left_hand', camera_id)
                right_hand_crop_dir = osp.join(local_root, date, video_id, crop_save_exp_name, 'right_hand', camera_id)
                os.makedirs(left_hand_crop_dir, exist_ok=True)
                os.makedirs(right_hand_crop_dir, exist_ok=True)

                bg = img_batch[f_idx][c_idx].copy()

                # crop info
                
                (right_min_h, right_max_h, right_min_w, right_max_w) = pos[0]
                right_min_h, right_max_h, right_min_w, right_max_w, right_mid_h, right_mid_w = apply_factor_and_margin(right_min_h, right_max_h, right_min_w, right_max_w, crop_factor, margin_size, MAX_HEIGHT, MAX_WIDTH)

                right_crop_info = [right_min_h, right_max_h, right_min_w, right_max_w, right_mid_h, right_mid_w]
                right_crop_info_batch[frame_id][camera_id] = right_crop_info
                
                # right_crop_info_path = osp.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl')
                # with open(right_crop_info_path, 'wb') as f:
                    # pickle.dump(right_crop_info, f)

                
                (left_min_h, left_max_h, left_min_w, left_max_w) = pos[1]
                left_min_h, left_max_h, left_min_w, left_max_w, left_mid_h, left_mid_w = apply_factor_and_margin(left_min_h, left_max_h, left_min_w, left_max_w, crop_factor, margin_size, MAX_HEIGHT, MAX_WIDTH)

                left_crop_info = [left_min_h, left_max_h, left_min_w, left_max_w, left_mid_h, left_mid_w]
                left_crop_info_batch[frame_id][camera_id] = left_crop_info
                
                # left_crop_info_path = osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl')
                # with open(left_crop_info_path, 'wb') as f:
                #     pickle.dump(left_crop_info, f)

                right_crop = crop(bg, right_min_h, right_max_h, right_min_w, right_max_w, left_min_h, left_max_h, left_min_w, left_max_w)
                right_crop_path = osp.join(right_hand_crop_dir, camera_id + '_' + frame_id + '.png')
                cv2.imwrite(right_crop_path, right_crop)

                # left hand
                left_crop = crop(bg, left_min_h, left_max_h, left_min_w, left_max_w, right_min_h, right_max_h, right_min_w, right_max_w)
                left_crop_path = osp.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png')
                cv2.imwrite(left_crop_path, left_crop)
    
    right_crop_info_dir = osp.join(local_root, date, video_id, crop_info_save_exp_name, 'right_hand')
    os.makedirs(right_crop_info_dir, exist_ok=True)
    left_crop_info_dir = osp.join(local_root, date, video_id, crop_info_save_exp_name, 'left_hand')
    os.makedirs(left_crop_info_dir, exist_ok=True)
    
    right_crop_info_path = osp.join(right_crop_info_dir, represent_frame_id + '_crop_info.pkl')
    with open(right_crop_info_path, 'wb') as f:
        pickle.dump(right_crop_info_batch, f)
    left_crop_info_path = osp.join(left_crop_info_dir, represent_frame_id + '_crop_info.pkl')
    with open(left_crop_info_path, 'wb') as f:
        pickle.dump(left_crop_info_batch, f)
        
def crop_from_init(root, date, video_id: str, camera_list: str, crop_factor = 1, margin_size = 0):
    '''
    手动标注第一二帧，从src中读取标注数据。
    crop一只手时，将另一只手的bbox所在区域全部置为黑。
    TODO: pos的shape有问题，要改
    '''
    frame_list = [str(i).zfill(5) for i in range(1, 3)]

    for camera_idx, camera_id in enumerate(camera_list):

        left_hand_crop_dir = osp.join(root, date, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = osp.join(root, date, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)

        for frame_idx, frame_id in enumerate(frame_list):

            pos = read_init_crop(video_id, camera_id, frame_id)
            crop_hand_from_pos(video_id, camera_id, frame_id, pos, crop_factor, margin_size)

def crop_from_mask_v2(root, video_id: str, from_exp_name, camera_list: str, frame_list: list[str], rgb_batch, crop_factor = 1, margin_size = 50):

    # for camera_id in camera_list:
    #     left_hand_crop_dir = os.path.join(root, video_id, 'crop', 'left_hand', camera_id)
    #     right_hand_crop_dir = os.path.join(root, video_id, 'crop', 'right_hand', camera_id)
    #     os.makedirs(left_hand_crop_dir, exist_ok=True)
    #     os.makedirs(right_hand_crop_dir, exist_ok=True)

    seg, downsample_factor = get_downsampled_seg_infos_batch_v2(video_id, from_exp_name, frame_list, camera_list)
    right_hand_seg = np.where(seg == 1, 1, 0).astype(np.uint8)
    left_hand_seg = np.where(seg == 2, 1, 0).astype(np.uint8)

    right_hand_valid_camera_dict = {}
    left_hand_valid_camera_dict = {}
    for frame in frame_list:
        right_hand_valid_camera_dict[frame] = []
        left_hand_valid_camera_dict[frame] = []

    for camera_idx, camera_id in enumerate(camera_list):
        left_hand_crop_dir = os.path.join(root, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = os.path.join(root, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)

        for frame_idx, frame_id in tqdm(enumerate(frame_list)):
            mask_right_hand = right_hand_seg[frame_idx, camera_idx]
            mask_left_hand = left_hand_seg[frame_idx, camera_idx]

            # 读入图片
            img = load_bg_img(video_id, camera_id, frame_id)

            # H = 3000
            # W = 4096
            (H, W) = img.shape[:2]
            MAX_WIDTH = W - 1
            MAX_HEIGHT = H - 1

            # upsample mask
            mask_right_hand = cv2.resize(mask_right_hand, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_left_hand = cv2.resize(mask_left_hand, (W, H), interpolation=cv2.INTER_NEAREST)

            # 以下为右手
            mask_right_hand_idx = np.nonzero(mask_right_hand)
            if len(mask_right_hand_idx[0]) != 0:
                right_hand_valid_camera_dict[frame_id].append(camera_id)

                right_minh = np.min(mask_right_hand_idx[0])
                right_maxh = np.max(mask_right_hand_idx[0])
                right_minw = np.min(mask_right_hand_idx[1])
                right_maxw = np.max(mask_right_hand_idx[1])
                right_midh = (right_maxh + right_minh) // 2
                right_midw = (right_maxw + right_minw) // 2
                right_meanh = np.mean(mask_right_hand_idx[0])
                right_meanw = np.mean(mask_right_hand_idx[1])

                right_length = int(max(right_maxh - right_minh, right_maxw - right_minw) * crop_factor)
                half_length = right_length // 2
                right_min_y = max(0, right_midh - half_length - margin_size)
                right_max_y = min(MAX_HEIGHT, right_midh + half_length + margin_size)
                right_min_x = max(0, right_midw - half_length - margin_size)
                right_max_x = min(MAX_WIDTH, right_midw + half_length + margin_size)

                # 将左手变成白色
                right_crop = img.copy()
                right_crop[mask_left_hand == 1] = 255.0

                # 居中
                right_crop = right_crop[right_min_y: right_max_y, right_min_x: right_max_x]
                right_crop_info = [right_min_y, right_max_y, right_min_x, right_max_x, right_meanh, right_meanw]

                with open(os.path.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                    pickle.dump(right_crop_info, f)

                cv2.imwrite(os.path.join(right_hand_crop_dir, camera_id + '_' +frame_id + '.png'), right_crop)

            # 以下为左手
            mask_left_hand_idx = np.nonzero(mask_left_hand)
            if len(mask_left_hand_idx[0]) != 0:
                left_hand_valid_camera_dict[frame_id].append(camera_id)

                left_minh = np.min(mask_left_hand_idx[0])
                left_maxh = np.max(mask_left_hand_idx[0])
                left_minw = np.min(mask_left_hand_idx[1])
                left_maxw = np.max(mask_left_hand_idx[1])
                left_midh = (left_maxh + left_minh) // 2
                left_midw = (left_maxw + left_minw) // 2
                left_meanh = np.mean(mask_left_hand_idx[0])
                left_meanw = np.mean(mask_left_hand_idx[1])

                left_length = max(left_maxh - left_minh, left_maxw - left_minw)
                half_length = left_length // 2
                left_min_y = max(0, left_midh - half_length - margin_size)
                left_max_y = min(MAX_HEIGHT, left_midh + half_length + margin_size)
                left_min_x = max(0, left_midw - half_length - margin_size)
                left_max_x = min(MAX_WIDTH, left_midw + half_length + margin_size)

                # 将右手变成白色
                left_crop = img.copy()
                left_crop[mask_right_hand == 1] = 255.0

                # 居中
                left_crop = left_crop[left_min_y: left_max_y, left_min_x: left_max_x]
                left_crop_info = [left_min_y, left_max_y, left_min_x, left_max_x, left_meanh, left_meanw]

                with open(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                    pickle.dump(left_crop_info, f)

                cv2.imwrite(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png'), left_crop)

    # save valid and invalid info
    right_hand_invalid_camera_dict = {}
    left_hand_invalid_camera_dict = {}
    for frame in frame_list:
        right_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in right_hand_valid_camera_dict[frame]]
        left_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in left_hand_valid_camera_dict[frame]]

    log_path = os.path.join(root, video_id, 'crop', 'log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
        log['right_hand_valid_camera_dict'].update(right_hand_valid_camera_dict)
        log['left_hand_valid_camera_dict'].update(left_hand_valid_camera_dict)
        log['right_hand_invalid_camera_dict'].update(right_hand_invalid_camera_dict)
        log['left_hand_invalid_camera_dict'].update(left_hand_invalid_camera_dict)
    else:
        log = {}
        log['right_hand_valid_camera_dict'] = right_hand_valid_camera_dict
        log['left_hand_valid_camera_dict'] = left_hand_valid_camera_dict
        log['right_hand_invalid_camera_dict'] = right_hand_invalid_camera_dict
        log['left_hand_invalid_camera_dict'] = left_hand_invalid_camera_dict

    with open(log_path, 'w') as f:
        json.dump(log, f)

    return right_hand_valid_camera_dict, left_hand_valid_camera_dict, right_hand_invalid_camera_dict, left_hand_invalid_camera_dict

def crop_from_mask_v3(root, video_id: str, from_exp_name, camera_list: str, frame_list: list[str], rgb_batch, crop_factor = 1, margin_size = 50):
    '''
    增加了rgb_batch，可以不用从本地再读图片。
    '''
    # for camera_id in camera_list:
    #     left_hand_crop_dir = os.path.join(root, video_id, 'crop', 'left_hand', camera_id)
    #     right_hand_crop_dir = os.path.join(root, video_id, 'crop', 'right_hand', camera_id)
    #     os.makedirs(left_hand_crop_dir, exist_ok=True)
    #     os.makedirs(right_hand_crop_dir, exist_ok=True)

    seg, downsample_factor = get_downsampled_seg_infos_batch_v2(video_id, from_exp_name, frame_list, camera_list)
    right_hand_seg = np.where(seg == 1, 1, 0).astype(np.uint8)
    left_hand_seg = np.where(seg == 2, 1, 0).astype(np.uint8)

    right_hand_valid_camera_dict = {}
    left_hand_valid_camera_dict = {}
    for frame in frame_list:
        right_hand_valid_camera_dict[frame] = []
        left_hand_valid_camera_dict[frame] = []

    for camera_idx, camera_id in enumerate(camera_list):
        left_hand_crop_dir = os.path.join(root, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = os.path.join(root, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)

        for frame_idx, frame_id in tqdm(enumerate(frame_list)):
            mask_right_hand = right_hand_seg[frame_idx, camera_idx]
            mask_left_hand = left_hand_seg[frame_idx, camera_idx]

            # 读入图片
            # img = rgb_batch[int(frame_id)-1,camera_idx].copy()
            # img = rgb_batch[int(frame_id)-1][camera_idx].copy()
            img = rgb_batch[frame_idx][camera_idx].copy()
            # img = load_bg_img(video_id, camera_id, frame_id)

            # H = 3000
            # W = 4096
            (H, W) = img.shape[:2]
            MAX_WIDTH = W - 1
            MAX_HEIGHT = H - 1

            # upsample mask
            mask_right_hand = cv2.resize(mask_right_hand, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_left_hand = cv2.resize(mask_left_hand, (W, H), interpolation=cv2.INTER_NEAREST)

            # 以下为右手
            mask_right_hand_idx = np.nonzero(mask_right_hand)
            if len(mask_right_hand_idx[0]) != 0:
                right_hand_valid_camera_dict[frame_id].append(camera_id)

                right_minh = np.min(mask_right_hand_idx[0])
                right_maxh = np.max(mask_right_hand_idx[0])
                right_minw = np.min(mask_right_hand_idx[1])
                right_maxw = np.max(mask_right_hand_idx[1])
                right_midh = (right_maxh + right_minh) // 2
                right_midw = (right_maxw + right_minw) // 2
                right_meanh = np.mean(mask_right_hand_idx[0])
                right_meanw = np.mean(mask_right_hand_idx[1])

                right_length = int(max(right_maxh - right_minh, right_maxw - right_minw) * crop_factor)
                half_length = right_length // 2
                right_min_y = max(0, right_midh - half_length - margin_size)
                right_max_y = min(MAX_HEIGHT, right_midh + half_length + margin_size)
                right_min_x = max(0, right_midw - half_length - margin_size)
                right_max_x = min(MAX_WIDTH, right_midw + half_length + margin_size)

                # 将左手变成白色
                right_crop = img.copy()
                right_crop[mask_left_hand == 1] = 255.0

                # 居中
                right_crop = right_crop[right_min_y: right_max_y, right_min_x: right_max_x]
                right_crop_info = [right_min_y, right_max_y, right_min_x, right_max_x, right_meanh, right_meanw]

                with open(os.path.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                    pickle.dump(right_crop_info, f)

                cv2.imwrite(os.path.join(right_hand_crop_dir, camera_id + '_' +frame_id + '.png'), right_crop)

            # 以下为左手
            mask_left_hand_idx = np.nonzero(mask_left_hand)
            if len(mask_left_hand_idx[0]) != 0:
                left_hand_valid_camera_dict[frame_id].append(camera_id)

                left_minh = np.min(mask_left_hand_idx[0])
                left_maxh = np.max(mask_left_hand_idx[0])
                left_minw = np.min(mask_left_hand_idx[1])
                left_maxw = np.max(mask_left_hand_idx[1])
                left_midh = (left_maxh + left_minh) // 2
                left_midw = (left_maxw + left_minw) // 2
                left_meanh = np.mean(mask_left_hand_idx[0])
                left_meanw = np.mean(mask_left_hand_idx[1])

                left_length = max(left_maxh - left_minh, left_maxw - left_minw)
                half_length = left_length // 2
                left_min_y = max(0, left_midh - half_length - margin_size)
                left_max_y = min(MAX_HEIGHT, left_midh + half_length + margin_size)
                left_min_x = max(0, left_midw - half_length - margin_size)
                left_max_x = min(MAX_WIDTH, left_midw + half_length + margin_size)

                # 将右手变成白色
                left_crop = img.copy()
                left_crop[mask_right_hand == 1] = 255.0

                # 居中
                left_crop = left_crop[left_min_y: left_max_y, left_min_x: left_max_x]
                left_crop_info = [left_min_y, left_max_y, left_min_x, left_max_x, left_meanh, left_meanw]

                with open(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                    pickle.dump(left_crop_info, f)

                cv2.imwrite(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png'), left_crop)

    # save valid and invalid info
    right_hand_invalid_camera_dict = {}
    left_hand_invalid_camera_dict = {}
    for frame in frame_list:
        right_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in right_hand_valid_camera_dict[frame]]
        left_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in left_hand_valid_camera_dict[frame]]

    log_path = os.path.join(root, video_id, 'crop', 'log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
        log['right_hand_valid_camera_dict'].update(right_hand_valid_camera_dict)
        log['left_hand_valid_camera_dict'].update(left_hand_valid_camera_dict)
        log['right_hand_invalid_camera_dict'].update(right_hand_invalid_camera_dict)
        log['left_hand_invalid_camera_dict'].update(left_hand_invalid_camera_dict)
    else:
        log = {}
        log['right_hand_valid_camera_dict'] = right_hand_valid_camera_dict
        log['left_hand_valid_camera_dict'] = left_hand_valid_camera_dict
        log['right_hand_invalid_camera_dict'] = right_hand_invalid_camera_dict
        log['left_hand_invalid_camera_dict'] = left_hand_invalid_camera_dict

    with open(log_path, 'w') as f:
        json.dump(log, f)

    return right_hand_valid_camera_dict, left_hand_valid_camera_dict, right_hand_invalid_camera_dict, left_hand_invalid_camera_dict

def crop_from_mask_v3_acc_batch(root, local_root, date, video_id: str, from_exp_name, camera_list: str, represent_frame_id, frame_list: list[str], BATCH_SIZE, rgb_batch, crop_factor = 1, margin_size = 50):
    '''
    增加了rgb_batch，可以不用从本地再读图片。
    每一个batch的文件分左右手写在一起。
    '''

    seg, downsample_factor = get_downsampled_seg_infos_batch_v2_acc_batch(root, date, video_id, from_exp_name, frame_list, camera_list, represent_frame_id=represent_frame_id)
    right_hand_seg = np.where(seg == 1, 1, 0).astype(np.uint8)
    left_hand_seg = np.where(seg == 2, 1, 0).astype(np.uint8)

    right_hand_valid_camera_dict = {}
    left_hand_valid_camera_dict = {}
    for frame in frame_list:
        right_hand_valid_camera_dict[frame] = []
        left_hand_valid_camera_dict[frame] = []

    # represent_relation = cal_represent_frame_list(BATCH_SIZE, frame_list)
    # represent_keys = list(represent_relation.keys())
    # assert len(represent_keys) == 1
    # represent_frame_id = represent_keys[0]

    left_crop_info_batch = {}
    left_crop_batch = {}
    right_crop_info_batch = {}
    right_crop_batch = {}

    for camera_idx, camera_id in enumerate(camera_list):
        left_hand_crop_dir = os.path.join(local_root, date, video_id, 'crop', 'left_hand', camera_id)
        right_hand_crop_dir = os.path.join(local_root, date, video_id, 'crop', 'right_hand', camera_id)
        os.makedirs(left_hand_crop_dir, exist_ok=True)
        os.makedirs(right_hand_crop_dir, exist_ok=True)
        
        for frame_idx, frame_id in tqdm(enumerate(frame_list)):
            mask_right_hand = right_hand_seg[frame_idx, camera_idx]
            mask_left_hand = left_hand_seg[frame_idx, camera_idx]

            # 读入图片
            img = rgb_batch[frame_idx][camera_idx].copy()

            # H = 3000
            # W = 4096
            (H, W) = img.shape[:2]
            MAX_WIDTH = W - 1
            MAX_HEIGHT = H - 1

            # upsample mask
            mask_right_hand = cv2.resize(mask_right_hand, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_left_hand = cv2.resize(mask_left_hand, (W, H), interpolation=cv2.INTER_NEAREST)

            # 以下为右手
            mask_right_hand_idx = np.nonzero(mask_right_hand)
            if len(mask_right_hand_idx[0]) != 0:
                right_hand_valid_camera_dict[frame_id].append(camera_id)

                right_minh = np.min(mask_right_hand_idx[0])
                right_maxh = np.max(mask_right_hand_idx[0])
                right_minw = np.min(mask_right_hand_idx[1])
                right_maxw = np.max(mask_right_hand_idx[1])
                right_midh = (right_maxh + right_minh) // 2
                right_midw = (right_maxw + right_minw) // 2
                right_meanh = np.mean(mask_right_hand_idx[0])
                right_meanw = np.mean(mask_right_hand_idx[1])

                right_length = int(max(right_maxh - right_minh, right_maxw - right_minw) * crop_factor)
                half_length = right_length // 2
                right_min_y = max(0, right_midh - half_length - margin_size)
                right_max_y = min(MAX_HEIGHT, right_midh + half_length + margin_size)
                right_min_x = max(0, right_midw - half_length - margin_size)
                right_max_x = min(MAX_WIDTH, right_midw + half_length + margin_size)

                # 将左手变成白色
                right_crop = img.copy()
                right_crop[mask_left_hand == 1] = 255.0

                # 居中
                right_crop = right_crop[right_min_y: right_max_y, right_min_x: right_max_x]
                right_crop_info = [right_min_y, right_max_y, right_min_x, right_max_x, right_meanh, right_meanw]

                if frame_id not in right_crop_info_batch:
                    right_crop_info_batch[frame_id] = {}
                right_crop_info_batch[frame_id][camera_id] = right_crop_info

                if frame_id not in right_crop_batch:
                    right_crop_batch[frame_id] = {}
                right_crop_batch[frame_id][camera_id] = right_crop

                # with open(os.path.join(right_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                #     pickle.dump(right_crop_info, f)

                cv2.imwrite(os.path.join(right_hand_crop_dir, camera_id + '_' +frame_id + '.png'), right_crop)

            # 以下为左手
            mask_left_hand_idx = np.nonzero(mask_left_hand)
            if len(mask_left_hand_idx[0]) != 0:
                left_hand_valid_camera_dict[frame_id].append(camera_id)

                left_minh = np.min(mask_left_hand_idx[0])
                left_maxh = np.max(mask_left_hand_idx[0])
                left_minw = np.min(mask_left_hand_idx[1])
                left_maxw = np.max(mask_left_hand_idx[1])
                left_midh = (left_maxh + left_minh) // 2
                left_midw = (left_maxw + left_minw) // 2
                left_meanh = np.mean(mask_left_hand_idx[0])
                left_meanw = np.mean(mask_left_hand_idx[1])

                left_length = max(left_maxh - left_minh, left_maxw - left_minw)
                half_length = left_length // 2
                left_min_y = max(0, left_midh - half_length - margin_size)
                left_max_y = min(MAX_HEIGHT, left_midh + half_length + margin_size)
                left_min_x = max(0, left_midw - half_length - margin_size)
                left_max_x = min(MAX_WIDTH, left_midw + half_length + margin_size)

                # 将右手变成白色
                left_crop = img.copy()
                left_crop[mask_right_hand == 1] = 255.0

                # 居中
                left_crop = left_crop[left_min_y: left_max_y, left_min_x: left_max_x]
                left_crop_info = [left_min_y, left_max_y, left_min_x, left_max_x, left_meanh, left_meanw]

                if frame_id not in left_crop_info_batch:
                    left_crop_info_batch[frame_id] = {}
                left_crop_info_batch[frame_id][camera_id] = left_crop_info

                if frame_id not in left_crop_batch:
                    left_crop_batch[frame_id] = {}
                left_crop_batch[frame_id][camera_id] = left_crop

                # with open(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '_crop_info.pkl'), 'wb') as f:
                #     pickle.dump(left_crop_info, f)

                cv2.imwrite(os.path.join(left_hand_crop_dir, camera_id + '_' + frame_id + '.png'), left_crop)


    # save valid and invalid info
    right_hand_invalid_camera_dict = {}
    left_hand_invalid_camera_dict = {}
    for frame in frame_list:
        right_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in right_hand_valid_camera_dict[frame]]
        left_hand_invalid_camera_dict[frame] = [camera for camera in camera_list if camera not in left_hand_valid_camera_dict[frame]]

    left_hand_crop_dir = os.path.join(local_root, date, video_id, 'crop_batch', 'left_hand')
    right_hand_crop_dir = os.path.join(local_root, date, video_id, 'crop_batch', 'right_hand')
    os.makedirs(left_hand_crop_dir, exist_ok=True)
    os.makedirs(right_hand_crop_dir, exist_ok=True)

    with open(os.path.join(right_hand_crop_dir, represent_frame_id + '_crop_info.pkl'), 'wb') as f:
        pickle.dump(right_crop_info_batch, f)
    with open(os.path.join(right_hand_crop_dir, represent_frame_id + '_crop.pkl'), 'wb') as f:
        pickle.dump(right_crop_batch, f)
        
    with open(os.path.join(left_hand_crop_dir, represent_frame_id + '_crop_info.pkl'), 'wb') as f:
        pickle.dump(left_crop_info_batch, f)
    with open(os.path.join(left_hand_crop_dir, represent_frame_id + '_crop.pkl'), 'wb') as f:
        pickle.dump(left_crop_batch, f)

    # 寻思这log也没人看啊
    # log_path = os.path.join(root, video_id, 'crop', 'log.json')
    # if os.path.exists(log_path):
    #     with open(log_path, 'r') as f:
    #         log = json.load(f)
    #     log['right_hand_valid_camera_dict'].update(right_hand_valid_camera_dict)
    #     log['left_hand_valid_camera_dict'].update(left_hand_valid_camera_dict)
    #     log['right_hand_invalid_camera_dict'].update(right_hand_invalid_camera_dict)
    #     log['left_hand_invalid_camera_dict'].update(left_hand_invalid_camera_dict)
    # else:
    #     log = {}
    #     log['right_hand_valid_camera_dict'] = right_hand_valid_camera_dict
    #     log['left_hand_valid_camera_dict'] = left_hand_valid_camera_dict
    #     log['right_hand_invalid_camera_dict'] = right_hand_invalid_camera_dict
    #     log['left_hand_invalid_camera_dict'] = left_hand_invalid_camera_dict

    # with open(log_path, 'w') as f:
    #     json.dump(log, f)

    return right_hand_valid_camera_dict, left_hand_valid_camera_dict, right_hand_invalid_camera_dict, left_hand_invalid_camera_dict

if __name__ == "__main__":
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    # camera_list = ['22139911']

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()
    video_id = args.video_id
    root = '/share/datasets/HOI-mocap'
    date = video_id[:8]

    img_dir = osp.join(root, date, video_id, 'imgs', camera_list[0]) # 默认每个视角的frame数相同
    assert os.path.isdir(img_dir)
    img_filename_list = list(scandir(img_dir, 'png'))
    frame_list = []
    for img_filename in img_filename_list:
        frame_id = img_filename[-9:-4]
        frame_list.append(frame_id)

    procs = []
    for camera_id in camera_list:
        args = (video_id, [camera_id], frame_list)
        proc = mlp.Process(target=crop_from_mask, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()

if __name__ == '__main__':
    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139916', '22139946']
    frame_list = [str(i).zfill(5) for i in range(1,3)]
    crop_from_init('20230904_01', camera_list, frame_list)