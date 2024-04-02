import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def filter_iamges_via_pixel_values(data_dir):
    images_lis = sorted(glob(os.path.join(data_dir, 'image/*.png'))) ## images lis ##
    n_images = len(images_lis)
    images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 255.0
    print(f"images_np: {images_np.shape}")
    # nn_frames x res x res x 3 #
    images_np = 1. - images_np
    has_density_values = (np.sum(images_np, axis=-1) > 0.7).astype(np.float32)
    has_density_values = np.sum(np.sum(has_density_values, axis=-1), axis=-1)
    tot_res_nns = float(images_np.shape[1] * images_np.shape[2])
    has_density_ratio = has_density_values / tot_res_nns ### has density ratio and ratio # 
    print(f"has_density_values: {has_density_values.shape}")
    paried_has_density_ratio_list = [(i_fr, has_density_ratio[i_fr].item()) for i_fr in range(has_density_ratio.shape[0])]
    paried_has_density_ratio_list = sorted(paried_has_density_ratio_list, key=lambda ii: ii[1], reverse=True)
    mid_rnk_value = len(paried_has_density_ratio_list) // 4
    print(f"mid value of the density ratio")
    print(paried_has_density_ratio_list[mid_rnk_value])
    iamge_idx = paried_has_density_ratio_list[mid_rnk_value][0]
    print(f"iamge idx: {images_lis[iamge_idx]}")
    print(paried_has_density_ratio_list[:mid_rnk_value])
    tot_selected_img_idx_list = [ii[0] for ii in paried_has_density_ratio_list[:mid_rnk_value]]
    tot_selected_img_idx_list =sorted(tot_selected_img_idx_list)
    print(len(tot_selected_img_idx_list))
    # print(tot_selected_img_idx_list[54])
    print(tot_selected_img_idx_list)
    
    

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        
        self.selected_img_idxes_list = [0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 35, 36, 42, 43, 44, 48, 49, 50, 51, 55, 56, 57, 61, 62, 63, 69, 84, 90, 91, 92, 96, 97]
        # self.selected_img_idxes_list = [0, 1, 5, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 28, 29, 35, 36, 37, 40, 41, 70, 71, 79, 82, 83, 84, 85, 92, 93, 96, 97, 98, 99, 105, 106, 107, 110, 111, 112, 113, 118, 119, 120, 121, 124, 125, 133, 134, 135, 139, 174, 175, 176, 177, 180, 188, 189, 190, 191, 194, 195]
        
        self.selected_img_idxes_list = [0, 1, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 36, 40, 41, 70, 71, 78, 82, 83, 84, 85, 90, 91, 92, 93, 96, 97]
        
        self.selected_img_idxes_list = [0, 1, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21, 22, 23, 26, 27, 36, 40, 41, 70, 71, 78, 82, 83, 84, 85, 90, 91, 92, 93, 96, 97, 98, 99, 104, 105, 106, 107, 110, 111, 112, 113, 118, 119, 120, 121, 124, 125, 134, 135, 139, 174, 175, 176, 177, 180, 181, 182, 183, 188, 189, 190, 191, 194, 195]
        
        self.selected_img_idxes_list = [0, 1, 6, 7, 8, 9, 12, 13, 14, 20, 21, 22, 23, 26, 27, 70, 78, 83, 84, 85, 91, 92, 93, 96, 97, 98, 99, 105, 106, 107, 110, 111, 112, 113, 119, 120, 121, 124, 125, 175, 176, 181, 182, 188, 189, 190, 191, 194, 195]
        # or the timestep to the dataset instance ## # selected img idxes list #
        self.selected_img_idxes = np.array(self.selected_img_idxes_list).astype(np.int32)
        
        
        
       

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        ## camera outside sphere ## 
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        # camera_dict = np.load("/home/xueyi/diffsim/NeuS/public_data/dtu_scan24/cameras_sphere.npz")
        self.camera_dict = camera_dict # rendr camera dict #
        # render camera dict # # number of pixels in the views -> very thin geometry is not useful 
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        
        # iamges_lis # and the images_lis and the images_lis #
        # self.images_lis = self.images_lis[:1] # totoal views and poses of the camera; # and select cameras for rendering #
        
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        
        
        self.selected_img_idxes_list = list(range(self.images_np.shape[0]))
        self.selected_img_idxes = np.array(self.selected_img_idxes_list).astype(np.int32)
        
        self.images_np = self.images_np[self.selected_img_idxes] ## get selected iamges_np #
        
        ### if we deal with the backgound carefully ### ### get 
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        self.images_np = self.images_np[self.selected_img_idxes]
        self.images_np = 1. - self.images_np ### 
        
        
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        
        # self.masks_lis = self.masks_lis[:1]
        
        try:
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
            self.masks_np = self.masks_np[self.selected_img_idxes]
        except:
            self.masks_np = self.images_np.copy()


        
        


        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        # for idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, self.world_mats_np)):
        for idx  in self.selected_img_idxes_list:
            scale_mat = self.scale_mats_np[idx]
            world_mat = self.world_mats_np[idx]
            
            if "hand" in self.data_dir:
                intrinsics = np.eye(4)
                fov = 512. / 2. # * 2
                res = 512.
                intrinsics[:3, :3] = np.array([
                    [fov, 0, 0.5* res], # res #
                    [0, fov, 0.5* res], # res #
                    [0, 0, 1]
                ], dtype=np.float32)
                pose = camera_dict['camera_mat_%d' % idx].astype(np.float32)
            else:
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = load_K_Rt_from_P(None, P)
                
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        ### images, masks, 
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3] #
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3] #
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4] # optimize sdf field # rigid model hand
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        
        self.n_images = self.images.size(0)

        print('Load data: End')
        
    def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
        i, j = torch.meshgrid( # meshgrid #
            torch.linspace(0, W-1, W, device=c2w.device),
            torch.linspace(0, H-1, H, device=c2w.device))
        i = i.t().float()
        j = j.t().float()
        if mode == 'lefttop':
            pass
        elif mode == 'center':
            i, j = i+0.5, j+0.5
        elif mode == 'random':
            i = i+torch.rand_like(i)
            j = j+torch.rand_like(j)
        else:
            raise NotImplementedError

        if flip_x:
            i = i.flip((1,))
        if flip_y:
            j = j.flip((0,))
        if inverse_y:
            dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
        else:
            dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,3].expand(rays_d.shape)
        return rays_o, rays_d

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        
        ##### previous method #####
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # # p = torch.stack([pixels_x, pixels_y, -1. * torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        ##### previous method #####
        
        fov = 512.; res = 512.
        K = np.array([
            [fov, 0, 0.5* res],
            [0, fov, 0.5* res],
            [0, 0, 1]
        ], dtype=np.float32)
        K = torch.from_numpy(K).float().cuda()
        
        
        # ### `center` mode ### #
        c2w = self.pose_all[img_idx]
        pixels_x, pixels_y = pixels_x+0.5, pixels_y+0.5
        
        dirs = torch.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -torch.ones_like(pixels_x)], -1)
        rays_v = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
        rays_o = c2w[:3,3].expand(rays_v.shape)
        # dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # # p = torch.stack([pixels_x, pixels_y, -1. * torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        # rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        # rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        
        
        ##### previous method #####
        # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        # # p = torch.stack([pixels_x, pixels_y, -1. * torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        # p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        # rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        # rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        ##### previous method #####
        
        fov = 512.; res = 512.
        K = np.array([
            [fov, 0, 0.5* res],
            [0, fov, 0.5* res],
            [0, 0, 1]
        ], dtype=np.float32)
        K = torch.from_numpy(K).float().cuda()
        
        
        # ### `center` mode ### #
        c2w = self.pose_all[img_idx]
        pixels_x, pixels_y = pixels_x+0.5, pixels_y+0.5
        
        dirs = torch.stack([(pixels_x-K[0][2])/K[0][0], -(pixels_y-K[1][2])/K[1][1], -torch.ones_like(pixels_x)], -1)
        rays_v = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
        rays_o = c2w[:3,3].expand(rays_v.shape)
        
        
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    ## iamge_at ##
    def image_at(self, idx, resolution_level):
        if self.selected_img_idxes_list is not None:
            img = cv.imread(self.images_lis[self.selected_img_idxes_list[idx]])
        else:
            img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


if __name__=='__main__':
    data_dir = "/data/datasets/genn/diffsim/diffredmax/save_res/goal_optimize_model_hand_sphere_test_obj_type_active_nfr_10_view_divide_0.5_n_views_7_three_planes_False_recon_dvgo_new_Nposes_7_routine_2"
    data_dir = "/data/datasets/genn/diffsim/neus/public_data/hand_test"
    data_dir = "/data2/datasets/diffsim/neus/public_data/hand_test_routine_2"
    data_dir = "/data2/datasets/diffsim/neus/public_data/hand_test_routine_2_light_color"
    filter_iamges_via_pixel_values(data_dir=data_dir)
