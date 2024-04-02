'''
读取pkl获得每一个视频的帧的数量，然后将video视频转换为一张张图片。

将一个batch的图片压成一个mp4视频，编号以第一帧为主。

example:
python utils/video2img.py --root_dir /share/datasets/HOI-mocap/20230904 --video_id 20230904_01
'''
import traceback
import os
import sys
sys.path.append('.')
import cv2
from tqdm import tqdm
import argparse
import pickle
import multiprocessing as mlp
from utils.hoi_io import get_valid_video_list

def divide_mp4(video_path, sub_video_dir, num_frame, original_num_frame, cnt2frame_id_dict, res_prefix='', res_suffix=''):
    os.makedirs(sub_video_dir, exist_ok=True)
    assert os.path.exists(video_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # print(fps, W, H)

    suc = cap.isOpened()
    BATCH_SIZE = 20

    cnt = 0 # 代表在原视频里的第{cnt}帧
    represent_frame_id = str(1).zfill(5)
    img_list = []
    while True:
        suc, img = cap.read()
        if not suc:
            break
        cnt += 1
        frame_id = cnt2frame_id_dict.get(cnt, None) # 匹配后的顺序中的第{frame_id}帧
        if frame_id is not None:
            if int(frame_id) % BATCH_SIZE == 3:

                # 写入sub_video
                fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
                sub_video_save_path = os.path.join(sub_video_dir, res_prefix + represent_frame_id + res_suffix + '.mp4')
                videoWriter = cv2.VideoWriter(sub_video_save_path, fourcc2, fps, (W, H))
                for img in img_list:
                    videoWriter.write(img)
                videoWriter.release()

                #更新写一次写入需要的数值
                img_list = []
                represent_frame_id = frame_id

            img_list.append(img)

    # 保存最后一次
    fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
    sub_video_save_path = os.path.join(sub_video_dir, res_prefix + represent_frame_id + res_suffix + '.mp4')
    videoWriter = cv2.VideoWriter(sub_video_save_path, fourcc2, fps, (W, H))
    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()

    if cnt != original_num_frame:
        print(f'video: {video_path} cnt: {cnt}, original_num_frame: {original_num_frame}')
    # assert cnt == original_num_frame, f'video: {video_path} cnt: {cnt}, original_num_frame: {original_num_frame}'

    cap.release()


if __name__ == "__main__":
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']

    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', required=True, type=str)
    # parser.add_argument('--video_id', required=True, type=str)
    args = parser.parse_args()

    # root_dir = args.root_dir
    # video_id = args.video_id
    
    date_list = ['20231015']
    
    for date in date_list:
        # date = '20231002'
        root_dir = f'/share/datasets/HOI-mocap/{date}'

        # video_list = [f'{date}_{str(i).zfill(3)}' for i in range(1, 54)]

        # dir_list = os.listdir(root_dir)
        # video_list = [dir for dir in dir_list if dir != 'camera_params' and 'cali' not in dir and not dir.endswith('txt')]
        # video_list.sort()

        video_list = get_valid_video_list(date, remove_hand=False)
        # video_list = [id for id in video_list if 'hand' in id]

        print(video_list)

        for video_id in tqdm(video_list):
            try:
                assert os.path.exists(root_dir)
                video_dir = os.path.join(root_dir, video_id)
                assert os.path.exists(video_dir)

                metadata_dir = os.path.join('/share/hlyang/results', date, video_id, 'metadata')

                # 部分数据出错，没有metadata，直接跳过
                if not os.path.exists(metadata_dir) or len(os.listdir(metadata_dir)) == 0:
                    print(f'{video_id}部分数据出错，没有metadata')
                    continue

                sub_video_root = os.path.join('/share/hlyang/results', date, video_id, 'sub_video')

                # 跳过已经处理过的
                # if os.path.exists(sub_video):
                #     continue

                os.makedirs(sub_video_root, exist_ok=True)

                procs = []

                for camera_id in camera_list:
                    metadata_path = os.path.join(metadata_dir, camera_id + '.pkl')
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    num_frame = metadata['num_frame']
                    original_num_frame = metadata['original_num_frame']
                    cnt2frame_id_dict = metadata['cnt2frame_id_dict']

                    video_path = os.path.join(video_dir, 'rgb', camera_id + '.mp4')
                    sub_video_dir = os.path.join(sub_video_root, camera_id)
                    os.makedirs(sub_video_dir, exist_ok=True)

                    # mp42img(video_path, img_dir, num_frame, res_prefix = camera_id + '_')
                    args = (video_path, sub_video_dir, num_frame, original_num_frame, cnt2frame_id_dict, camera_id + '_')
                    proc = mlp.Process(target=divide_mp4, args=args)
                    proc.start()
                    procs.append(proc)

                for i in range(len(procs)):
                    procs[i].join()
            except Exception as err:
                traceback.print_exc()
                print(err)
                print(f'{video_id} failed!')