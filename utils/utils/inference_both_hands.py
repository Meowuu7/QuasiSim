'''
TODO：根据device再进行细分

python utils/inference_both_hands.py --video_id 20230715_15
'''

from mmpose.apis import MMPoseInferencer
import os.path as osp
import os
from tqdm import tqdm
import argparse
import numpy as np
import multiprocessing as mlp
import torch

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def hand_detect(img_path_list, result_dir, gpu):

    device = torch.device(gpu)  # 选择要使用的 GPU 设备
    torch.cuda.set_device(device)

    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer('hand')

    # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer(img_path_list, out_dir=result_dir)
    for path in tqdm(img_path_list):
        next(result_generator)

def handle_divide_by_gpu_capacity(video_id, img_filename_list, save_dir, gpu):
    MAX_GPU_CAPACITY = 5
    num_img = len(img_filename_list)
    num_img_per_process = np.ceil(num_img / MAX_GPU_CAPACITY).astype(np.uint32)

    procs = []
    for i in range(MAX_GPU_CAPACITY):
        start_frame_idx = i * num_img_per_process
        end_frame_idx = min(start_frame_idx + num_img_per_process, num_img)  # 不包含
        img_filename_list_sub = img_filename_list[start_frame_idx:end_frame_idx]

        args = (img_filename_list_sub, save_dir, gpu)
        proc = mlp.Process(target=hand_detect, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()

def handle_divide_by_num_gpu(video_id, right_hand_bool):
    if right_hand_bool:
        # hand_dir = os.path.join('/share/hlyang/results', video_id, 'crop_imgs_right_hand')
        # save_dir = os.path.join('/share/hlyang/results', video_id, 'mmpose_right_hand')
        hand_dir = os.path.join('/share/hlyang/results', video_id, 'crop', 'right_hand')
        save_dir = os.path.join('/share/hlyang/results', video_id, 'mmpose', 'right_hand')
    else:
        # hand_dir = os.path.join('/share/hlyang/results', video_id, 'crop_imgs_left_hand')
        # save_dir = os.path.join('/share/hlyang/results', video_id, 'mmpose_left_hand')
        hand_dir = os.path.join('/share/hlyang/results', video_id, 'crop', 'left_hand')
        save_dir = os.path.join('/share/hlyang/results', video_id, 'mmpose', 'left_hand')

    img_filename_list = list(scandir(hand_dir, suffix='png', recursive=True, full_path=True))
    num_img = len(img_filename_list)

    GPU_list = [1, 2, 3, 4, 6, 7]
    NUM_GPU = len(GPU_list)
    num_img_per_gpu = np.ceil(num_img / NUM_GPU).astype(np.uint32)

    procs = []
    for i in range(NUM_GPU):
        start_frame_idx = i * num_img_per_gpu
        end_frame_idx = min(start_frame_idx + num_img_per_gpu, num_img)  # 不包含
        img_filename_list_sub = img_filename_list[start_frame_idx:end_frame_idx]

        # print(img_filename_list_sub)

        args = (video_id, img_filename_list_sub, save_dir, GPU_list[i])
        proc = mlp.Process(target=handle_divide_by_gpu_capacity, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_id', required=True, type=str)
    # args = parser.parse_args()
    # video_id = args.video_id

    video_list = [f'20230818_0{i}_old' for i in ('4', '5', '6', '7')]

    for video_id in video_list:
        procs = []

        args = (video_id, True)
        proc = mlp.Process(target=handle_divide_by_num_gpu, args=args)
        proc.start()
        procs.append(proc)

        args = (video_id, False)
        proc = mlp.Process(target=handle_divide_by_num_gpu, args=args)
        proc.start()
        procs.append(proc)

        for i in range(len(procs)):
            procs[i].join()