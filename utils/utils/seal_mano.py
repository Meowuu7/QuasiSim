import sys
sys.path.append('.')
import numpy as np
import torch
from manopth.manopth.manolayer import ManoLayer
import open3d as o3d

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)

def seal_mano_mesh(v3d, faces, is_rhand):
    # v3d: B, 778, 3
    # faces: 1538, 3
    # output: v3d(B, 779, 3); faces (1554, 3)

    seal_faces = torch.LongTensor(np.array(SEAL_FACES_R)).to(faces.device)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    centers = v3d[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    sealed_vertices = torch.cat((v3d, centers), dim=1)
    faces = torch.cat((faces, seal_faces), dim=0)
    return sealed_vertices, faces

if __name__ == "__main__":
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
    
    right_hand_mesh = o3d.geometry.TriangleMesh()
    right_hand_pcd = o3d.geometry.PointCloud()
    right_hand_pcd.points = o3d.utility.Vector3dVector(right_hand_sealed_vertices.numpy()[0])
    right_hand_mesh.vertices = right_hand_pcd.points
    right_hand_mesh.triangles = o3d.utility.Vector3iVector(right_hand_faces)
    left_hand_mesh = o3d.geometry.TriangleMesh()
    left_hand_pcd = o3d.geometry.PointCloud()
    left_hand_pcd.points = o3d.utility.Vector3dVector(left_hand_sealed_vertices.numpy()[0])
    left_hand_mesh.vertices = left_hand_pcd.points
    left_hand_mesh.triangles = o3d.utility.Vector3iVector(left_hand_faces)
    
    voxel_size = 0.001
    right_hand_vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(right_hand_mesh, voxel_size)
    right_voxel_list = right_hand_vox.get_voxels()
    right_grid_indexs = np.stack([voxel.grid_index for voxel in right_voxel_list])
    left_hand_vox = o3d.geometry.VoxelGrid.create_from_triangle_mesh(left_hand_mesh, voxel_size)
    left_voxel_list = left_hand_vox.get_voxels()
    left_grid_indexs = np.stack([voxel.grid_index for voxel in left_voxel_list])
    
    right_set = {tuple(row) for row in right_grid_indexs}
    left_set = {tuple(row) for row in left_grid_indexs}
    intersection_set = right_set.intersection(left_set)
    intersection_set = np.array(list(intersection_set))
    
    num_right_hand_vox = right_grid_indexs.shape[0]
    num_left_hand_vox = left_grid_indexs.shape[0]
    num_intersection_vox = intersection_set.shape[0]
    print(num_right_hand_vox, num_left_hand_vox, num_intersection_vox)
    print()
    
    o3d.io.write_triangle_mesh("/home/hlyang/HOI/HOI/tmp/right_T_pose.obj", right_hand_mesh)
    o3d.io.write_triangle_mesh("/home/hlyang/HOI/HOI/tmp/left_T_pose.obj", left_hand_mesh)
    