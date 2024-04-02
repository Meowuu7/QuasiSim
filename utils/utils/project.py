import os
import sys
sys.path.append('.')
import torch
import math

def project_point_to_face(point, face_vertices):
    # 计算面的法线
    edge1 = face_vertices[1] - face_vertices[0]
    edge2 = face_vertices[2] - face_vertices[0]
    face_normal = torch.cross(edge1, edge2)
    
    # 计算面上的任意一点（这里使用面的第一个顶点）
    face_point = face_vertices[0]
    
    # 计算点到面的向量
    point_to_face = face_point - point
    
    # 计算点到面的投影点
    projection = point + (torch.dot(point_to_face, face_normal) / torch.dot(face_normal, face_normal)) * face_normal
    
    return projection

def project_point_to_face_batch(point, face_vertices_batch):
    # 计算面的法线
    edge1_batch = face_vertices_batch[:, 1, :] - face_vertices_batch[:, 0, :]
    edge2_batch = face_vertices_batch[:, 2, :] - face_vertices_batch[:, 0, :]
    face_normal_batch = torch.cross(edge1_batch, edge2_batch, dim=-1)
    
    # 计算面上的任意一点（这里使用面的第一个顶点）
    face_point_batch = face_vertices_batch[:, 0, :]
    
    # 计算点到面的向量
    point_to_face_batch = face_point_batch - point.unsqueeze(0)
    
    # 计算点到面的投影点
    projection_batch = point.unsqueeze(0) + (torch.sum(point_to_face_batch * face_normal_batch, dim=-1) / torch.sum(face_normal_batch * face_normal_batch, dim=-1)).unsqueeze(1) * face_normal_batch
    
    return projection_batch

def is_point_in_face(point, face_vertices):
    # 计算三角形的法向量
    edge1 = face_vertices[1] - face_vertices[0]
    edge2 = face_vertices[2] - face_vertices[1]
    edge3 = face_vertices[0] - face_vertices[2]

    # 计算待判断点到三角形顶点的向量
    to_vertex1 = point - face_vertices[0]
    to_vertex2 = point - face_vertices[1]
    to_vertex3 = point - face_vertices[2]

    # 计算点到三角形顶点的法向量
    normal1 = torch.cross(edge1, to_vertex1)
    normal2 = torch.cross(edge2, to_vertex2)
    normal3 = torch.cross(edge3, to_vertex3)  # 注意取反以保持方向一致

    # 检查法向量是否同向
    is_inside = (torch.dot(normal1, normal2) >= 0) and (torch.dot(normal2, normal3) >= 0) and (torch.dot(normal3, normal1) >= 0)
    
    return is_inside
    
def is_point_in_face_batch(point_batch, face_vertices_batch):
    # 计算三角形的法向量
    edge1_batch = face_vertices_batch[:, 1, :] - face_vertices_batch[:, 0, :]
    edge2_batch = face_vertices_batch[:, 2, :] - face_vertices_batch[:, 1, :]
    edge3_batch = face_vertices_batch[:, 0, :] - face_vertices_batch[:, 2, :]

    # 计算待判断点到三角形顶点的向量
    to_vertex1_batch = point_batch - face_vertices_batch[:, 0, :]
    to_vertex2_batch = point_batch - face_vertices_batch[:, 1, :]
    to_vertex3_batch = point_batch - face_vertices_batch[:, 2, :]

    # 计算点到三角形顶点的法向量
    normal1_batch = torch.cross(edge1_batch, to_vertex1_batch)
    normal2_batch = torch.cross(edge2_batch, to_vertex2_batch)
    normal3_batch = torch.cross(edge3_batch, to_vertex3_batch)  # 注意取反以保持方向一致

    # 检查法向量是否同向
    judge1 = torch.sum(normal1_batch * normal2_batch, dim=-1) >= 0
    judge2 = torch.sum(normal2_batch * normal3_batch, dim=-1) >= 0
    judge3 = torch.sum(normal3_batch * normal1_batch, dim=-1) >= 0
    is_inside_batch = judge1 & judge2 & judge3
    
    return is_inside_batch

def is_point_inside_faces_projection(vertices, faces, point):
    
    mask_list = []
    
    for face_indices in faces:
        face_vertices = vertices[face_indices]
        projection = project_point_to_face(point, face_vertices)
        mask_list.append(is_point_in_face(projection, face_vertices))
    
    mask = torch.tensor(mask_list)
    return mask

def cal_dist_from_point_to_face(point, face_vertices):
    # 计算面的法线
    edge1 = face_vertices[1] - face_vertices[0]
    edge2 = face_vertices[2] - face_vertices[0]
    face_normal = torch.cross(edge1, edge2)
    
    # 计算面上的任意一点（这里使用面的第一个顶点）
    face_point = face_vertices[0]
    
    # 计算点到面的向量
    point_to_face = face_point - point
    
    # 计算点到面的垂线长度
    dist = torch.abs(torch.dot(point_to_face, face_normal) / torch.norm(face_normal))
    return dist

def cal_dist_from_point_to_face_batch(point, face_vertices_batch):
    # 计算面的法线
    edge1_batch = face_vertices_batch[:, 1] - face_vertices_batch[:, 0]
    edge2_batch = face_vertices_batch[:, 2] - face_vertices_batch[:, 0]
    face_normal_batch = torch.cross(edge1_batch, edge2_batch, dim=-1)
    
    # 标准化face_normal_batch
    face_normal_batch = face_normal_batch / torch.norm(face_normal_batch, dim=-1).unsqueeze(1)
    
    # 计算面上的任意一点（这里使用面的第一个顶点）
    face_point_batch = face_vertices_batch[:, 0, :]
    
    # 计算点到面的向量
    point_to_face_batch = face_point_batch - point.unsqueeze(0)
    
    # 计算点到面的垂线长度
    dist_batch = torch.sum(point_to_face_batch*face_normal_batch, dim=-1)
    dist_batch_abs = torch.abs(dist_batch)
    
    # assert torch.any(dist_batch_abs == torch.inf) == False, 'dist_batch_abs should not contain inf'
    
    # 计算垂点
    point_stroke_batch = point.unsqueeze(0) + dist_batch.unsqueeze(1) * face_normal_batch
    
    return dist_batch_abs, point_stroke_batch

def get_min_dist_batch(vertices, faces, point):
    # TODO 考虑mask全为False的情况
    
    mask_batch = is_point_inside_faces_projection(vertices, faces, point)
    
    dist_batch = []
    for idx, mask in enumerate(mask_batch):
        if mask == True:
            face_vertices = vertices[faces[idx]]
            dist = cal_dist_from_point_to_face(point, face_vertices)
            dist_batch.append(dist)
        else:
            dist_batch.append(torch.inf)
    dist_batch = torch.tensor(dist_batch)
    min_dist, min_idx = torch.min(dist_batch, 0)
    return min_dist, min_idx

def get_vertical_direction_from_point_to_face(point, face_vertices):
    projection = project_point_to_face(point, face_vertices)
    # 检查投影点是否与原始点相等
    if torch.all(torch.eq(projection, point)):
        # 如果相等，返回法向量作为方向
        edge1 = face_vertices[1] - face_vertices[0]
        edge2 = face_vertices[2] - face_vertices[0]
        face_normal = torch.cross(edge1, edge2)
        direction = face_normal / torch.norm(face_normal)
    else:
        vertical = projection - point
        direction = vertical / torch.norm(vertical)
    return direction

def get_intersection_points(point, direction, point_stroke_batch, dist_batch):
    perpendicular_batch = point_stroke_batch - point.unsqueeze(0)
    projection_len_batch = torch.sum(perpendicular_batch * direction.unsqueeze(0), dim=-1) / torch.norm(perpendicular_batch, dim=-1)
    projection_factor_batch = dist_batch / projection_len_batch
    
    point_to_intersection_points = projection_factor_batch.unsqueeze(1) * direction.unsqueeze(0)
    len_of_point_to_intersection_points = torch.norm(point_to_intersection_points, dim=-1)
    intersection_points = point.unsqueeze(0) + point_to_intersection_points
    return intersection_points, len_of_point_to_intersection_points

def get_nearest_intersection_point(vertices, faces, points, valid_vert_indices, dist_threshold):
    # TODO：考虑找不到direction的情况
    # TODO: 考虑dist_batch的值全部为torch.inf的情况
    device = vertices.device
    
    intersection_point_list = []
    intersection_point_valid_mask = []
    
    for point in points:
    
        projection_batch = project_point_to_face_batch(point, vertices[faces])
        mask_batch = is_point_in_face_batch(projection_batch, vertices[faces])
        
        # dist_batch = []
        # for idx, mask in enumerate(mask_batch):
        #     if mask == True:
        #         face_vertices = vertices[faces[idx]]
        #         dist = cal_dist_from_point_to_face(point, face_vertices)
        #         dist_batch.append(dist)
        #     else:
        #         dist_batch.append(torch.inf)
        # dist_batch = torch.tensor(dist_batch)
        
        
        # TODO 有可能所选的点都不在面内
        dist_batch, point_stroke_batch = cal_dist_from_point_to_face_batch(point, vertices[faces])
        # dist_batch中mask_batch为False的索引值赋为inf
        dist_batch[~mask_batch] = torch.inf
        assert torch.any(dist_batch != torch.inf)
        min_dist, min_idx = torch.min(dist_batch, 0)
        direction = get_vertical_direction_from_point_to_face(point, vertices[faces[min_idx]])
        
        print(point_stroke_batch[min_idx])
        
        # TODO 交点求的有问题
        # intersection_point_batch = point.unsqueeze(0) + dist_batch.unsqueeze(1) * direction
        intersection_point_batch, len_of_point_to_intersection_points = get_intersection_points(point, direction, point_stroke_batch, dist_batch)
        print(intersection_point_batch[min_idx])
        is_intersection_point_in_faces_batch = is_point_in_face_batch(intersection_point_batch, vertices[faces])
        
        # print(intersection_point_batch[is_intersection_point_in_faces_batch])
        
        
        face_valid_mask = []
        for idx, face_indices in enumerate(faces):
            face_valid_mask.append(torch.all(torch.isin(face_indices, valid_vert_indices)))
        face_valid_mask = torch.tensor(face_valid_mask).to(device)
        
        # print(is_intersection_point_in_faces_batch.sum(), face_valid_mask[is_intersection_point_in_faces_batch], len_of_point_to_intersection_points[is_intersection_point_in_faces_batch])
        
        # print((torch.norm(dist_batch.unsqueeze(1) * direction, dim=-1) <= dist_threshold).sum())
        
        judge = face_valid_mask & is_intersection_point_in_faces_batch & (len_of_point_to_intersection_points <= dist_threshold)
        if torch.any(judge) == False:
            intersection_point_list.append(torch.tensor([0, 0, 0], device=device))
            intersection_point_valid_mask.append(False)
            continue

        # 好像上面的判断多余了，如果全是torch.inf，那么min_dist也是torch.inf
        
        # dist_batch中面非法的索引值赋为inf
        dist_batch[~face_valid_mask] = torch.inf
        
        # dist_batch中交点不在面内部的索引值赋为inf
        dist_batch[~is_intersection_point_in_faces_batch] = torch.inf
        
        # dist_batch中值超过阈值的索引值赋为inf
        dist_batch[len_of_point_to_intersection_points > dist_threshold] = torch.inf
        
        min_dist, min_idx = torch.min(dist_batch, 0)
        intersection_point = intersection_point_batch[min_idx]
        # print(face_valid_mask[min_idx], is_intersection_point_in_faces_batch[min_idx], len_of_point_to_intersection_points[min_idx] <= dist_threshold)
        
        intersection_point_list.append(intersection_point)
        intersection_point_valid_mask.append(True)
            
    #     intersection_points = []
    #     dist_batch = []
    #     face_indices_batch = []
    #     for face_indices in faces:
    #         if torch.all(torch.isin(face_indices, valid_vert_indices)):
    #             face_vertices = vertices[face_indices]
    #             dist = cal_dist_from_point_to_face(point, face_vertices)
    #             intersection_point = point + dist * direction
                
    #             # 判断intersection_point是否在三角形内部
    #             if is_point_in_face(intersection_point, face_vertices) and torch.norm(dist * direction) < dist_threshold:
    #                 intersection_points.append(intersection_point)
    #                 dist_batch.append(torch.norm(dist * direction))
    #                 face_indices_batch.append(face_indices)
        
    #     # assert len(intersection_points) > 0, 'len(intersection_points) should be greater than 0'
    #     if len(intersection_points) == 0:
    #         intersection_point_list.append(torch.tensor([0, 0, 0]))
    #         intersection_point_valid_mask.append(False)
    #         continue
        
    #     dist_batch = torch.tensor(dist_batch)
    #     min_dist, min_idx = torch.min(dist_batch, 0)
    #     intersection_point = intersection_points[min_idx]
    #     face_indices = face_indices_batch[min_idx]
        
    #     intersection_point_list.append(intersection_point)
    #     intersection_point_valid_mask.append(True)
    
    intersection_point_list = torch.stack(intersection_point_list)
    intersection_point_valid_mask = torch.tensor(intersection_point_valid_mask)
    
    return intersection_point_list, intersection_point_valid_mask

def get_nearest_intersection_point_batch(vertices_batch, faces, points_batch, valid_vert_indices, dist_threshold):
    num_frames = vertices_batch.shape[0]
    
    intersection_point_batch = []
    intersection_point_valid_mask_batch = []
    
    for i in range(num_frames):
        intersection_point_list, intersection_point_valid_mask = get_nearest_intersection_point(vertices_batch[i], faces, points_batch[i], valid_vert_indices, dist_threshold)
        intersection_point_batch.append(intersection_point_list)
        intersection_point_valid_mask_batch.append(intersection_point_valid_mask)
        print(intersection_point_list)
        print(intersection_point_valid_mask)
        print('----------------------------------')
        
    intersection_point_batch = torch.stack(intersection_point_batch)
    intersection_point_valid_mask_batch = torch.stack(intersection_point_valid_mask_batch)
    
    return intersection_point_batch, intersection_point_valid_mask_batch