'''
读取pkl获得每一个视频的帧的数量，然后将video视频转换为一张张图片。

TODO：是否要将图片压成一个npy文件？也许可以减少IO。re:千万别，太吃内存

example:
python utils/video2img.py --root_dir /share/datasets/HOI-mocap/20230904 --video_id 20230904_01
'''

import os
import sys
sys.path.append('.')
import cv2
from tqdm import tqdm
import argparse
import pickle
import multiprocessing as mlp
from utils.hoi_io import get_valid_video_list

def mp42img(video_path, img_dir, num_frame, original_num_frame, cnt2frame_id_dict, res_prefix='', res_suffix=''):
    os.makedirs(img_dir, exist_ok=True)
    assert os.path.exists(video_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(fps, W, H)

    suc = cap.isOpened()

    # for frame_cnt in tqdm(range(1, num_frame + 1)):
    #     suc, img = cap.read()
    #     assert suc
    #     cv2.imwrite(os.path.join(img_dir, res_prefix + str(frame_cnt).zfill(5) + res_suffix + ".png"), img)
    cnt = 0
    while True:
        suc, img = cap.read()
        if not suc:
            break
        cnt += 1
        frame_id = cnt2frame_id_dict.get(cnt, None)
        if frame_id is not None:
            cv2.imwrite(os.path.join(img_dir, res_prefix + frame_id + res_suffix + ".png"), img)
    assert cnt == original_num_frame, f'cnt: {cnt}, original_num_frame: {original_num_frame}'

    cap.release()

if __name__ == "__main__":
    # camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']

    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', required=True, type=str)
    # parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()

    # root_dir = args.root_dir
    # video_id = args.video_id
    date = '20231005'
    root_dir = f'/share/datasets/HOI-mocap/{date}'

    # video_list = [f'{date}_{str(i).zfill(3)}' for i in range(1, 54)]

    # dir_list = os.listdir(root_dir)
    # video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]
    # video_list.sort()

    video_list = get_valid_video_list(date)

    print(video_list)

    for video_id in tqdm(video_list):
        assert os.path.exists(root_dir)
        video_dir = os.path.join(root_dir, video_id)
        assert os.path.exists(video_dir)

        # metadata_dir = os.path.join('/share/hlyang/results', video_id, 'metadata')
        metadata_dir = os.path.join('/share/hlyang/results', date, video_id, 'metadata')

        # 部分数据出错，没有metadata，直接跳过
        if not os.path.exists(metadata_dir) or len(os.listdir(metadata_dir)) == 0:
            print(f'{video_id}部分数据出错，没有metadata')
            continue

        # img_root = os.path.join('/share/hlyang/results', video_id, 'imgs')
        img_root = os.path.join('/share/hlyang/results', date, video_id, 'imgs')


        # 跳过已经处理过的
        # if os.path.exists(img_root):
        #     continue

        os.makedirs(img_root, exist_ok=True)

        procs = []

        for camera_id in camera_list:
            metadata_path = os.path.join(metadata_dir, camera_id + '.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            num_frame = metadata['num_frame']
            original_num_frame = metadata['original_num_frame']
            cnt2frame_id_dict = metadata['cnt2frame_id_dict']

            video_path = os.path.join(video_dir, 'rgb', camera_id + '.mp4')
            img_dir = os.path.join(img_root, camera_id)
            os.makedirs(img_dir, exist_ok=True)

            # mp42img(video_path, img_dir, num_frame, res_prefix = camera_id + '_')
            args = (video_path, img_dir, num_frame, original_num_frame, cnt2frame_id_dict, camera_id + '_')
            proc = mlp.Process(target=mp42img, args=args)
            proc.start()
            procs.append(proc)

        for i in range(len(procs)):
            procs[i].join()