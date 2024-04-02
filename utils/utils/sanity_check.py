'''
按处理顺序检查文件完整性。

example:
python utils/sanity_check.py --video_id 20230715_15
'''

import os
from tqdm import tqdm
import argparse
import pickle
import multiprocessing as mlp

if __name__ == "__main__":
    # camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139913', '22139916', '22139946']
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_id', required=True, type=str)
    # args = parser.parse_args()
    # video_id = args.video_id

    video_list = [f'20230915_{str(i).zfill(3)}' for i in range(4,16)]
    for video_id in video_list:

        num_frame = None
        for camera_id in camera_list:
            metadata_path = os.path.join('/share/hlyang/results', video_id, 'metadata', camera_id+'.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                num_frame = data['num_frame']
            else:
                print('metadata sanity check failed!')
                exit(1)

        frame_list = [str(i+1).zfill(5) for i in range(num_frame)]

        video_res_dir = os.path.join('/share/hlyang/results', video_id)

        for camera_id in camera_list:
            for frame_id in frame_list:
                # imgs
                path = os.path.join(video_res_dir, 'imgs', camera_id, camera_id + '_' + frame_id + '.png')
                if not os.path.exists(path):
                    print('sanity check failed!', path)
                    exit(1)

                # results
                path = os.path.join(video_res_dir, 'fit_hand_joint_ransac_batch_by_squence', 'res', 'left_hand', f'hand_{frame_id}.pkl')
                if not os.path.exists(path):
                    print('sanity check failed!', path)
                path = os.path.join(video_res_dir, 'fit_hand_joint_ransac_batch_by_squence', 'res', 'right_hand', f'hand_{frame_id}.pkl')
                if not os.path.exists(path):
                    print('sanity check failed!', path)

                # results
                path = os.path.join(video_res_dir, 'get_world_mesh_from_mano_params', 'meshes', 'left_hand', f'hand_{frame_id}.obj')
                if not os.path.exists(path):
                    print('sanity check failed!', path)
                path = os.path.join(video_res_dir, 'get_world_mesh_from_mano_params', 'meshes', 'right_hand', f'hand_{frame_id}.obj')
                if not os.path.exists(path):
                    print('sanity check failed!', path)

                # anno_results
                # path = os.path.join(video_res_dir, 'anno_results', camera_id, camera_id + '_' + frame_id + '.npy')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)

                # crop
                # path = os.path.join(video_res_dir, 'crop_imgs_left_hand', camera_id, camera_id + '_' + frame_id + '_crop_info.pkl')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)
                # path = os.path.join(video_res_dir, 'crop_imgs_right_hand', camera_id, camera_id + '_' + frame_id + '_crop_info.pkl')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)
                # path = os.path.join(video_res_dir, 'crop_imgs_left_hand', camera_id, camera_id + '_' + frame_id + '.png')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)
                # path = os.path.join(video_res_dir, 'crop_imgs_right_hand', camera_id, camera_id + '_' + frame_id + '.png')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)

                # mmpose
                # path = os.path.join(video_res_dir, 'mmpose_left_hand', 'predictions', camera_id + '_' + frame_id + '.json')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)
                # path = os.path.join(video_res_dir, 'mmpose_right_hand', 'predictions', camera_id + '_' + frame_id + '.json')
                # if not os.path.exists(path):
                #     print('sanity check failed!', path)
                #     exit(1)