'''
读取obj文件，并且通过o3d将mesh进行downsample以减少计算量，指定将其downsample到某个顶点数量。

TODO: 在20230715_15中mask == 3是左手操控的物体，与后面的标准相反。此处暂时将左手操作的物体记作object1。

example:
python utils/downsample_mesh.py --object_src_path /share/datasets/HOI-mocap/object_models/0802-zsz-object004/object004_cm.obj --video_id 20230715_15 --object_label 1
'''

import open3d as o3d
import argparse
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_src_path', required=True, type=str)
    parser.add_argument('--video_id', required=True, type=str)
    parser.add_argument('--object_label', required=True, type=int)
    parser.add_argument('--num_verts', default=2000, required=False, type=int)
    args = parser.parse_args()

    object_src_path = args.object_src_path
    video_id = args.video_id
    object_label = args.object_label
    num_verts = args.num_verts
    
    assert os.path.exists(object_src_path)
    
    mesh = o3d.io.read_triangle_mesh(object_src_path)
    simplified_mesh = mesh.simplify_quadric_decimation(num_verts)  # 指定要保留的顶点数量
    
    save_dir = os.path.join('/share/hlyang/results', video_id, 'src')
    os.makedirs(save_dir, exist_ok=True)
    if object_label == 1:
        save_path = save_dir = os.path.join(save_dir, 'object1.obj')
    else:
        save_path = save_dir = os.path.join(save_dir, 'object2.obj')
    o3d.io.write_triangle_mesh(save_path, simplified_mesh)

# # 加载网格
# mesh = o3d.io.read_triangle_mesh('../src/Scan.obj')

# # 进行网格下采样
# simplified_mesh = mesh.simplify_quadric_decimation(2000)  # 指定要保留的顶点数量

# # 保存下采样后的网格
# o3d.io.write_triangle_mesh('../src/bottle.obj', simplified_mesh)