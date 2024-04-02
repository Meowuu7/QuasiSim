import os
import sys
sys.path.append('.')
from shutil import copy, rmtree
from os.path import join
from hoi_io2 import get_valid_video_list
from tqdm import tqdm

# if __name__ == '__main__':

#     date_list = ['20230923']

#     for date in date_list:
#         # video_list = [f'20230915_{str(i).zfill(3)}' for i in (4, 6, 8, 9, 10, 11, 12, 13, 15)]
#         # video_list = get_valid_video_list(date)
#         video_list = ['20230930_hand1']

#         # clean_dir_list = ['4_mesh_vis']
#         clean_dir_list = ['4_mesh_vis_old', 'crop', 'fake_mask_track', 'fake_mask_track_failed', 'fit_hand_joint_ransac_batch_by_squence', 'get_world_mesh_from_mano_params', 'joint_ransac_every_joint_triangulation', 'mask', 'mmpose']
#         for video_id in tqdm(video_list):
#             root = join('/share/hlyang/results', video_id)
#             for dir in clean_dir_list:
#                 path = join(root, dir)
#                 if os.path.exists(path):
#                     print('clean dir', path)
#                     rmtree(path)

if __name__ == '__main__':

    root = '/data2/hlyang/results'
    date_list = ['20231019']

    for date in date_list:
        # given_list = [f'20231013_{str(i).zfill(3)}' for i in range(132, 192)]
        video_list = get_valid_video_list(root, date, remove_hand=True)

        # clean_dir_list = ['4_mesh_vis']
        # clean_dir_list = ['crop', 'crop_batch', 'fake_mask_track', 'fit_hand_batch', 'joint_opt_batch', 'mask', 'mmpose', 'mmpose_batch', 'ransac_batch', 'world_mesh_batch']
        clean_dir_list = ['sub_video']
        for video_id in tqdm(video_list):
            root_video = join(root, date, video_id)
            for dir in clean_dir_list:
                path = join(root_video, dir)
                if os.path.exists(path):
                    print('clean dir', path)
                    rmtree(path)
