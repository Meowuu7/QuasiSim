import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from utils.hoi_io import render_from_mano_params, load_bg_img, cvt_multiview_imgs2mp4, load_mano_info_batch, get_camera_params
import torch
import numpy as np
import cv2
import pickle
from manopth.manopth.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras, PointLights, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, SoftPhongShader, TexturesVertex)
from tqdm import tqdm

def get_bbox_from_hand_params(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, enlarge_factor, device):
    '''
    return:
    bbox: shape: [num_frame, num_camera, 4] [min_h, max_h, min_w, max_w]
    '''
    device = torch.device(device)
    num_frame = len(frame_list)
    num_camera = len(camera_list)

    rendered_image_batch = render_from_mano_params(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, device)
    rendered_image_batch = rendered_image_batch.numpy()

    MAX_HEIGHT = 4095
    MAX_WIDTH = 2999

    # TODO：试试批处理能不能更快，似乎太大了后np.nonzero会卡住
    bbox_list = []
    for i in range(num_frame):
        bbox_list_camera = []
        for j in range(num_camera):
            non_white_pixels = np.all(rendered_image_batch[i, j, ...] != [1., 1., 1.], axis=-1)
            hand_idx = np.nonzero(non_white_pixels)
            min_h = np.min(hand_idx[0])
            max_h = np.max(hand_idx[0])
            mid_h = (min_h + max_h) // 2

            min_w = np.min(hand_idx[1])
            max_w = np.max(hand_idx[1])
            mid_w = (min_w + max_w) // 2
            
            length = int(max(max_h - min_h, max_w - min_w) * enlarge_factor)

            half_length = length // 2
            min_h_ = max(0, mid_h - half_length)
            max_h_ = min(MAX_WIDTH, mid_h + half_length)
            min_w_ = max(0, mid_w - half_length)
            max_w_ = min(MAX_HEIGHT, mid_w + half_length)
            # bbox_list_camera.append(torch.IntTensor([min_h_, max_h_, min_w_, max_w_]))

            # numpy 顺序 左上角x，左上角y，右下角x，右下角y
            # bbox_list_camera.append(np.array([min_h_, max_h_, min_w_, max_w_]))
            bbox_list_camera.append(np.array([min_w_, min_h_, max_w_, max_h_]))

        bbox_list_camera = np.stack(bbox_list_camera)
        bbox_list.append(bbox_list_camera)
    bbox_list = np.stack(bbox_list)
    return bbox_list

def get_bbox_from_hand_params_light_memory_use(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, enlarge_factor, device):
    '''
    TODO: 增加从文件读取shape参数
    '''
    
    hand_trans_batch, hand_pose_batch, _ = load_mano_info_batch(video_id, from_exp_name, frame_list, right_hand_bool)
    num_frame = len(frame_list)
    
    device = torch.device(device)
    
    calibration_info_path = os.path.join('/share/hlyang/results', video_id, 'src', 'calibration.json')
    assert os.path.join(calibration_info_path)
    R, R_inverse, T, K, focal_length, principal_point, image_size = get_camera_params(calibration_info_path, camera_list)
    
    use_pca = False
    ncomps = 45
    if right_hand_bool:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx=0)
    else:
        mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx=0)
        
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    camera = PerspectiveCameras(device=device, R=R_inverse, T=T, image_size=image_size, in_ndc=False, focal_length=-focal_length, principal_point=principal_point)
    raster_settings = RasterizationSettings(
        image_size=(3000, 4096),
        blur_radius=0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings), shader=SoftPhongShader(device=device, cameras=camera, lights=lights))
    faces_idx = mano_layer.th_faces.detach().clone().to(device)
    
    MAX_HEIGHT = 4095
    MAX_WIDTH = 2999
    
    bbox_list = []
    for i, frame in tqdm(enumerate(frame_list)):
        bbox_list_camera = []
        
        hand_pose = hand_pose_batch[i, ...]
        hand_trans = hand_trans_batch[i, ...]
        
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.unsqueeze(0)
        verts, _, _ = mano_layer(hand_pose)
        verts = verts.squeeze()
        verts = verts / 1000.0
        verts += hand_trans
        verts = verts.to(device)
        
        
        mesh = Meshes(verts=[verts], faces=[faces_idx])
        color = torch.ones(1, verts.size(0), 3, device=device)
        color[:, :, 2] = 255
        mesh.textures = TexturesVertex(verts_features=color)
        mesh = mesh.extend(R.shape[0])

        images = renderer(mesh)[..., :3].squeeze().cpu().numpy()
        
        for j, camera in enumerate(camera_list):
            non_white_pixels = np.all(images[j, ...] != [1., 1., 1.], axis=-1)
            hand_idx = np.nonzero(non_white_pixels)
            min_h = np.min(hand_idx[0])
            max_h = np.max(hand_idx[0])
            mid_h = (min_h + max_h) // 2

            min_w = np.min(hand_idx[1])
            max_w = np.max(hand_idx[1])
            mid_w = (min_w + max_w) // 2
            
            length = int(max(max_h - min_h, max_w - min_w) * enlarge_factor)

            half_length = length // 2
            min_h_ = max(0, mid_h - half_length)
            max_h_ = min(MAX_WIDTH, mid_h + half_length)
            min_w_ = max(0, mid_w - half_length)
            max_w_ = min(MAX_HEIGHT, mid_w + half_length)
            
            length = int(max(max_h - min_h, max_w - min_w) * enlarge_factor)
            
            half_length = length // 2
            min_h_ = max(0, mid_h - half_length)
            max_h_ = min(MAX_WIDTH, mid_h + half_length)
            min_w_ = max(0, mid_w - half_length)
            max_w_ = min(MAX_HEIGHT, mid_w + half_length)

            bbox_list_camera.append(np.array([min_w_, min_h_, max_w_, max_h_]))
        bbox_list_camera = np.stack(bbox_list_camera)
        bbox_list.append(bbox_list_camera)
    bbox_list = np.stack(bbox_list)
    return bbox_list
            
def draw_bbox_from_hand_params(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, enlarge_factor, resize_bool, resize_resolution, device):
    device = torch.device(device)
    # bbox = get_bbox_from_hand_params(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, enlarge_factor, device)
    bbox = get_bbox_from_hand_params_light_memory_use(video_id, from_exp_name, camera_list, frame_list, right_hand_bool, enlarge_factor, device)

    # 每一帧用前一帧的结果画bbox
    point_color = (0, 255, 0)
    thickness = 20
    lineType = 4
    
    for frame_idx, frame in enumerate(frame_list[1:]): # frame_idx从0开始，frame从1开始
        for camera_idx, camera in enumerate(camera_list):
            if right_hand_bool:
                save_dir = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'right_hand', camera)
            else:
                save_dir = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'left_hand', camera)
                
            os.makedirs(save_dir, exist_ok=True)
            bg = load_bg_img(video_id, camera, frame)
            cv2.rectangle(bg, bbox[frame_idx, camera_idx, 0:2], bbox[frame_idx, camera_idx, 2:4], point_color, thickness, lineType)
            if resize_bool:
                cv2.resize(bg, resize_resolution, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_dir, f'{camera}_{frame}.png'), bg)


if __name__ == '__main__':
    exp_name = 'get_bbox_from_hand_params'
    resize_bool = True
    resize_resolution = (1024, 750)
    factor = 1.4
    device = 7

    camera_list = ['22070938', '22139905', '22139909', '22139910', '22139911', '22139916', '22139946']

    video_id_list = [f'20230818_0{i}' for i in ('3', '4', '5', '6', '7')]
    for video_id in video_id_list:
        
        metadata_path = os.path.join('/share/hlyang/results', video_id, 'metadata', f'{camera_list[0]}.pkl')
        assert os.path.exists(metadata_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            start = 1
            end = metadata['num_frame']
        frame_list = [str(frame).zfill(5) for frame in range(start, end + 1)]
        
    # frame_list = [str(i).zfill(5) for i in range(1,5)]
    # video_id = '20230818_03'
    
        draw_bbox_from_hand_params(video_id, 'fit_hand_joint_ransac_batch_by_squence', camera_list, frame_list, True, factor, resize_bool, resize_resolution, device)
        
        img_dir = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'right_hand')
        save_video_path = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'right_hand', 'right_hand.mp4')
        cvt_multiview_imgs2mp4(img_dir, save_video_path, 1, camera_list, frame_list[1:])
        save_video_path = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'right_hand', 'right_hand_30fps.mp4')
        cvt_multiview_imgs2mp4(img_dir, save_video_path, 30, camera_list, frame_list[1:])
        
        draw_bbox_from_hand_params(video_id, 'fit_hand_joint_ransac_batch_by_squence', camera_list, frame_list, False, factor, resize_bool, resize_resolution, device)
        
        img_dir = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'left_hand')
        save_video_path = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'left_hand', 'left_hand.mp4')
        cvt_multiview_imgs2mp4(img_dir, save_video_path, 1, camera_list, frame_list[1:])
        save_video_path = os.path.join('/share/hlyang/results', video_id, exp_name, 'vis', 'left_hand', 'left_hand_30fps.mp4')
        cvt_multiview_imgs2mp4(img_dir, save_video_path, 30, camera_list, frame_list[1:])
