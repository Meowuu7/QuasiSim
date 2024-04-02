import os
import sys
sys.path.append('.')
import numpy as np
import trimesh
import cv2
import torch
from tqdm import tqdm

from prepare_2Dmask.utils.pyt3d_wrapper import Pyt3DWrapper
from prepare_2Dmask.utils.json_to_caminfo import json_to_caminfo
from prepare_2Dmask.utils.colors import FAKE_COLOR_LIST
from prepare_2Dmask.utils.visualization import render_HO_meshes

from utils.hoi_io2 import load_bg_imgs_with_resize

# try:
# import polyscope as ps
# ps.init()
# ps.set_ground_plane_mode("none")
# ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
# ps.set_screenshot_extension(".png")
# except:
#     pass

import sys
sys.path.append("./manopth")
from manopth.manopth.manolayer import ManoLayer

color = [
(0,191/255.0,255/255.0),
    (186/255.0,85/255.0,211/255.0),
    (255/255.0,81/255.0,81/255.0),
    (92/255.0,122/255.0,234/255.0),
    (255/255.0,138/255.0,174/255.0),
    (77/255.0,150/255.0,255/255.0),
    (192/255.0,237/255.0,166/255.0)
    #
]

def seal(v, f):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype=np.int32)
    center = (v[circle_v_id, :]).mean(0)

    # sealed_mesh = copy.copy(mesh_to_seal)
    v = np.vstack([v, center])
    center_v_id = v.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i - 1], circle_v_id[i], center_v_id]
        f = np.vstack([f, new_faces])
    return v, f, center

def get_mano_model(ncomps=45, side='right', flat_hand_mean=False,):
    # ncomps = 45 # mano root #
    batch_size = 1
    mano_model = ManoLayer(mano_root='manopth/mano/models', use_pca=False if ncomps == 45 else True, ncomps=ncomps, flat_hand_mean=flat_hand_mean, side=side, center_idx=0)
    return mano_model

def vis_predicted(root, nokov_root, video_id, camera_list, stg1_use_t, stg2_use_t, seed, st, predicted_info_fn, optimized_fn=None,  ws=60, device=0):
    date = video_id[:8]
    mano_model = get_mano_model(side='right')
    faces = mano_model.th_faces.squeeze(0).numpy()
    
    H_downsampled = 750
    W_downsampled = 1024    
    save_height = 3000
    save_width = 4096
    dowmsampled_factor = 4
    save_fps = 30
    save_height_view = save_height // dowmsampled_factor
    save_width_view = save_width // dowmsampled_factor

    ws = ws
    is_toch = False
    # predicted_info_data = np.load(predicted_info_fn, allow_pickle=True).item()
    if optimized_fn is not None:
        data = np.load(optimized_fn, allow_pickle=True).item()
        print(f"keys of optimized dict: {data.keys()}")
        optimized_out_hand_verts_woopt = data["bf_ct_verts"]
        optimized_out_hand_verts = optimized_out_hand_verts_woopt
    else:
        optimized_out_hand_verts = None

    data = np.load(predicted_info_fn, allow_pickle=True).item()

    try:
        targets = data['targets']
    except:
        targets = data['tot_gt_rhand_joints']

    outputs = data['outputs']
    if 'obj_verts' in data:
        obj_verts = data['obj_verts']
        obj_faces = data['obj_faces']
    elif 'tot_obj_pcs' in data:
        obj_verts = data['tot_obj_pcs'][0]
        obj_faces = data['template_obj_fs']
    tot_base_pts = data["tot_base_pts"][0]

    if 'tot_obj_rot' in data:
        tot_obj_rot = data['tot_obj_rot'][0]
        tot_obj_trans = data['tot_obj_transl'][0]
        obj_verts = np.matmul(obj_verts, tot_obj_rot) + tot_obj_trans.reshape(tot_obj_trans.shape[0], 1, 3)  # ws x nn_obj x 3 #

        outputs = np.matmul(outputs, tot_obj_rot) + tot_obj_trans.reshape(tot_obj_trans.shape[0], 1, 3)  # ws x nn_obj x 3 #

    # jts_radius = 0.01787
    jts_radius = 0.03378
    gray_color = (233 / 255., 241 / 255., 148 / 255.)

    camera_info_path = os.path.join(root, date, video_id, 'src', 'calibration.json')
    cam_info = json_to_caminfo(camera_info_path, camera_list=camera_list)

    device = torch.device(device)

    pyt3d_wrapper_dict = {}
    for camera in camera_list:
        pyt3d_wrapper_dict[camera] = Pyt3DWrapper(rasterization_image_size=(W_downsampled, H_downsampled), camera_image_size=cam_info[camera]["image_size"], use_fixed_cameras=True, intrin=cam_info[camera]["intrinsic"], extrin=cam_info[camera]["extrinsic"], device=device, colors=FAKE_COLOR_LIST, use_ambient_lights=False)

    # frame_list = [str(i).zfill(5) for i in range(1, ws+1)]
    frame_list = [str(i).zfill(5) for i in range(1+int(st), ws+int(st)+1)]

    rgb_batch = load_bg_imgs_with_resize(root, video_id, frame_list, camera_list, BATCH_SIZE=20, width=W_downsampled, height=H_downsampled)

    video_save_dir = os.path.join('/data3/hlyang/results/vis_dataset_denoise_test', date)
    os.makedirs(video_save_dir, exist_ok=True)
    video_save_path = os.path.join(video_save_dir, f"{video_id}_st_{st}_ws_{ws}_seed_{seed}_use_t_{stg1_use_t}_{stg2_use_t}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    videoWriter = cv2.VideoWriter(video_save_path, fourcc, save_fps, (save_width_view * 4, save_height_view * 3))

    maxx_ws = ws
    # skipp = 6
    skipp = 1
    iidx = 1
    tot_hand_verts_woopt = []
    for i_fr in tqdm(range(0, min(maxx_ws, optimized_out_hand_verts.shape[0]), skipp)):
        cur_base_pts = tot_base_pts

        if i_fr < obj_verts.shape[0]:
            cur_obj_verts = obj_verts[i_fr]
            cur_obj_faces = obj_faces


        if optimized_out_hand_verts is not None:
            sealed_v, seald_f, center_wopt = seal(optimized_out_hand_verts[i_fr], faces)

            # print(sealed_v.shape, seald_f.shape)
            hand_mesh = trimesh.Trimesh(vertices=sealed_v, faces=seald_f)
            # hand_mesh.export('/home/hlyang/HOI/HOI/tmp/hand_denoised.obj')
            # exit(1)
            # hand_mesh = ps.register_surface_mesh(f"cur_hand_mesh", sealed_v, seald_f, color=color[0 % len(color)])

        # print(cur_obj_verts.shape, cur_obj_faces.shape)
        obj_mesh = trimesh.Trimesh(vertices=cur_obj_verts, faces=cur_obj_faces)
        # obj_mesh = ps.register_surface_mesh(f"cur_object", cur_obj_verts, cur_obj_faces, color=gray_color)
        
        meshes = [hand_mesh, obj_mesh]
        frame = str(i_fr+1).zfill(5)
        saved_img = np.zeros((save_height_view * 3, save_width_view * 4, 3)).astype(np.uint8)
        
        for c_idx, camera in enumerate(camera_list):
            bg = rgb_batch[i_fr, c_idx, ...]
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            
            img = render_HO_meshes(pyt3d_wrapper_dict[camera], meshes, bg)
            img =cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (save_width_view, save_height_view))
            
            cv2.putText(img, f'{frame} {camera}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            saved_img[save_height_view*(c_idx//4):save_height_view*((c_idx//4)+1), save_width_view*(c_idx%4):save_width_view*((c_idx%4)+1)] = img
        
        videoWriter.write(saved_img)
        iidx += 1
        
    videoWriter.release()
        
    print(iidx-1)

if __name__=='__main__':
    root = '/data3/hlyang/results'
    upload_root = '/data2/HOI-mocap'
    camera_list = ['21218078', '22070938', '22139905', '22139906', '22139908', '22139909', '22139910', '22139911', '22139913', '22139914', '22139916', '22139946']
    cuda = 1
    
    video_id = '20231104_001'
    date = video_id[:8]
    stg1_use_t = '200'
    stg2_use_t = '200'
    seed = '0'
    st = '30'
    n_tag = '2'
    
    # predicted_info_fn = "./save_res/predicted_infos_sv_dict_seq_0_seed_110_tag_jts_spatial_t_200_hho__0_jts_spatial_t_200_multi_ntag_3.npy"
    # optimized_fn = "./save_res/optimized_infos_sv_dict_seq_0_seed_110_tag_jts_t_50_rep_arctic_st_100__0_jts_spatial_t_200_dist_thres_0.001_with_proj_False_wmaskanchors_multi_ntag_3.npy"
    predicted_info_fn = f'/data3/hlyang/results/denoise_test/{date}/{video_id}/predicted_infos_sv_dict_seq_0_seed_{seed}_tag_{video_id}_spatial_jts_t_{stg1_use_t}_st_{st}_hho__0_jts_spatial_t_{stg2_use_t}_multi_ntag_{n_tag}.npy'
    optimized_fn = f'/data3/hlyang/results/denoise_test/{date}/{video_id}/optimized_infos_sv_dict_seq_0_seed_{seed}_tag_{video_id}_spatial_jts_t_{stg1_use_t}_st_{st}_hho__0_jts_spatial_t_{stg2_use_t}_dist_thres_0.001_with_proj_False_wmaskanchors_multi_ntag_{n_tag}.npy'
    # ws = 60 
    ws = 30*int(n_tag) + 30
    vis_predicted(root, upload_root, video_id, camera_list, stg1_use_t, stg2_use_t, seed, st, predicted_info_fn, optimized_fn=optimized_fn, ws=ws, device=cuda)