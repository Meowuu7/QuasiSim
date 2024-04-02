import sys
sys.path.append('.')
import numpy as np
import open3d as o3d

def cal_num_intersection_vox(mesh_x, mesh_y, voxel_size=0.001):
    voxel_x = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_x, voxel_size)
    voxel_y = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_y, voxel_size)
    voxel_list_x = voxel_x.get_voxels()
    grid_indexs_x = np.stack([voxel.grid_index for voxel in voxel_list_x])
    voxel_list_y = voxel_y.get_voxels()
    grid_indexs_y = np.stack([voxel.grid_index for voxel in voxel_list_y])
    
    set_x = {tuple(row) for row in grid_indexs_x}
    set_y = {tuple(row) for row in grid_indexs_y}
    intersection_set = set_x.intersection(set_y)
    intersection_set = np.array(list(intersection_set))
    
    num_x = grid_indexs_x.shape[0]
    num_y = grid_indexs_y.shape[0]
    num_intersection = intersection_set.shape[0]
    return num_x, num_y, num_intersection

def construct_mesh_from_verts_and_faces(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    mesh.vertices = pcd.points
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

if __name__ == "__main__":
    from utils.seal_mano import seal_mano_mesh
    from manopth.manopth.manolayer import ManoLayer
    import torch
    
    use_pca = False
    ncomps = 45
    left_hand_mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='left', center_idx = 0)
    right_hand_mano_layer = ManoLayer(mano_root='./manopth/mano/models', use_pca=use_pca, ncomps=ncomps, side='right', center_idx = 0)
    
    right_hand_verts, _, _ = right_hand_mano_layer(torch.zeros(1, 48))
    right_hand_faces = right_hand_mano_layer.th_faces.detach()
    left_hand_verts, _, _ = left_hand_mano_layer(torch.zeros(1, 48))
    left_hand_faces = left_hand_mano_layer.th_faces.detach()
    
    right_hand_sealed_vertices, right_hand_faces = seal_mano_mesh(right_hand_verts / 1000.0, right_hand_faces, True)
    left_hand_sealed_vertices, left_hand_faces = seal_mano_mesh(left_hand_verts / 1000.0, left_hand_faces, False)
    
    # right_hand_mesh = trimesh.Trimesh(right_hand_sealed_vertices[0], right_hand_faces)
    # left_hand_mesh = trimesh.Trimesh(left_hand_sealed_vertices[0], left_hand_faces)
    
    # right_hand_vox = right_hand_mesh.voxelized(0.001)
    # left_hand_vox = left_hand_mesh.voxelized(0.001)
    
    # right_hand_mesh = o3d.geometry.TriangleMesh()
    # right_hand_pcd = o3d.geometry.PointCloud()
    # right_hand_pcd.points = o3d.utility.Vector3dVector(right_hand_sealed_vertices.numpy()[0])
    # right_hand_mesh.vertices = right_hand_pcd.points
    # right_hand_mesh.triangles = o3d.utility.Vector3iVector(right_hand_faces)
    # left_hand_mesh = o3d.geometry.TriangleMesh()
    # left_hand_pcd = o3d.geometry.PointCloud()
    # left_hand_pcd.points = o3d.utility.Vector3dVector(left_hand_sealed_vertices.numpy()[0])
    # left_hand_mesh.vertices = left_hand_pcd.points
    # left_hand_mesh.triangles = o3d.utility.Vector3iVector(left_hand_faces)
    
    right_hand_mesh = construct_mesh_from_verts_and_faces(right_hand_sealed_vertices.numpy()[0], right_hand_faces)
    left_hand_mesh = construct_mesh_from_verts_and_faces(left_hand_sealed_vertices.numpy()[0], left_hand_faces)
    
    # voxel_size = 0.001
    # right_hand_vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(right_hand_mesh, voxel_size)
    # right_voxel_list = right_hand_vox.get_voxels()
    # right_grid_indexs = np.stack([voxel.grid_index for voxel in right_voxel_list])
    # left_hand_vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(left_hand_mesh, voxel_size)
    # left_voxel_list = left_hand_vox.get_voxels()
    # left_grid_indexs = np.stack([voxel.grid_index for voxel in left_voxel_list])
    
    # right_set = {tuple(row) for row in right_grid_indexs}
    # left_set = {tuple(row) for row in left_grid_indexs}
    # intersection_set = right_set.intersection(left_set)
    # intersection_set = np.array(list(intersection_set))
    
    # num_right_hand_vox = right_grid_indexs.shape[0]
    # num_left_hand_vox = left_grid_indexs.shape[0]
    # num_intersection_vox = intersection_set.shape[0]
    # print(num_right_hand_vox, num_left_hand_vox, num_intersection_vox)
    # print()
    
    num_right_hand, num_left_hand, num_intersection = cal_num_intersection_vox(right_hand_mesh, left_hand_mesh)
    print(num_right_hand, num_left_hand, num_intersection)
    
    # o3d.io.write_triangle_mesh("/home/hlyang/HOI/HOI/tmp/right_T_pose.obj", right_hand_mesh)
    # o3d.io.write_triangle_mesh("/home/hlyang/HOI/HOI/tmp/left_T_pose.obj", left_hand_mesh)
    