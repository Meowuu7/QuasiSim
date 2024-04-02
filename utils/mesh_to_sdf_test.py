import os
import numpy as np
import taichi as ti

ti.init()


real = ti.f32


def get_sdf_spatial_grad():
    sdf_sv_fn = "/data/xueyi/diffsim/NeuS/init_box_mesh.npy"
    if not os.path.exists(sdf_sv_fn):
        sdf_sv_fn = "/home/xueyi/diffsim/NeuS/init_box_mesh.npy"
    sdf_sv = np.load(sdf_sv_fn, allow_pickle=True)
    scene_sdf = sdf_sv
    print(sdf_sv.shape)
    
    res = sdf_sv.shape[0]
    
    sdf_grad = np.zeros((res, res, res, 3), dtype=np.float32)
    
    
    # scene_sdf = ti.field(dtype=real, shape=(self.sdf_res, self.sdf_res, self.sdf_res))
    for i_x in range(res):
        print(f"Start processing i_x : {i_x}")
        for i_y in range(res):
            print(f"Start processing i_x : {i_x}, i_y : {i_y}")
            for i_z in range(res):
                cur_grad = np.zeros((3,), dtype=np.float32)
                
                if i_x > 0 and i_x < res - 1:
                    cur_grad[0] = (scene_sdf[i_x + 1, i_y, i_z] - scene_sdf[i_x - 1, i_y, i_z]) / 2.0
                elif i_x == 0:
                    cur_grad[0] = scene_sdf[i_x + 1, i_y, i_z] - scene_sdf[i_x, i_y, i_z]
                elif i_x == res - 1:
                    cur_grad[0] = scene_sdf[i_x, i_y, i_z] - scene_sdf[i_x - 1, i_y, i_z]
                
                if i_y > 0 and i_y < res - 1:
                    cur_grad[1] = (scene_sdf[i_x, i_y + 1, i_z] - scene_sdf[i_x, i_y - 1, i_z]) / 2.0
                elif i_y == 0:
                    cur_grad[1] = scene_sdf[i_x, i_y + 1, i_z] - scene_sdf[i_x, i_y, i_z]
                elif i_y == res - 1:
                    cur_grad[1] = scene_sdf[i_x, i_y, i_z] - scene_sdf[i_x, i_y - 1, i_z]
                
                if i_z > 0 and i_z < res - 1:
                    cur_grad[2] = (scene_sdf[i_x, i_y, i_z + 1] - scene_sdf[i_x, i_y, i_z - 1]) / 2.0
                elif i_z == 0:
                    cur_grad[2] = scene_sdf[i_x, i_y, i_z + 1] - scene_sdf[i_x, i_y, i_z]
                elif i_z == res - 1:
                    cur_grad[2] = scene_sdf[i_x, i_y, i_z] - scene_sdf[i_x, i_y, i_z - 1]
                
                sdf_grad[i_x, i_y, i_z, :] = cur_grad[:]
    
    sdf_grad_sv_fn = "/home/xueyi/diffsim/NeuS/init_box_mesh_sdf_grad.npy"
    np.save(sdf_grad_sv_fn, sdf_grad)

if __name__ == '__main__':
    get_sdf_spatial_grad() # 