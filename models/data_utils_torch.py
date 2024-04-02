"""Mesh data utilities."""
from re import I
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d  # pylint: disable=unused-import
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
import six
import os
import math
import torch
import models.fields as fields
# from options.options import opt
# from polygen_torch.
# from utils.constants import MASK_GRID_VALIE

try:
    from torch_cluster import fps
except:
    pass

MAX_RANGE = 0.1
MIN_RANGE = -0.1

# import open3d as o3d

def calculate_correspondences(last_mesh, bending_network, tot_timesteps, delta_bending):
  # iterate all the timesteps and get the bended 
  timestep_to_pts = {
    tot_timesteps - 1: last_mesh.detach().cpu().numpy()
  }
  if delta_bending:
    
    for i_ts in range(tot_timesteps - 1, -1, -1):
      if isinstance(bending_network, list):
        tot_offsets = []
        for i_bending, cur_bending_net in enumerate(bending_network):
          cur_def_pts = cur_bending_net(last_mesh, i_ts)
          tot_offsets.append(cur_def_pts - last_mesh)
        tot_offsets = torch.stack(tot_offsets, dim=0)
        tot_offsets = torch.sum(tot_offsets, dim=0)
        last_mesh = last_mesh + tot_offsets
      else:
        last_mesh = bending_network(last_mesh, i_ts)
      timestep_to_pts[i_ts - 1] = last_mesh.detach().cpu().numpy()
  elif isinstance(bending_network, fields.BendingNetworkRigidTrans):
    for i_ts in range(tot_timesteps - 1, -1, -1):
      last_mesh = bending_network.forward_delta(last_mesh, i_ts)
      timestep_to_pts[i_ts - 1] = last_mesh.detach().cpu().numpy()
  else:
    raise ValueError(f"the function is designed for delta bending")
  return timestep_to_pts
  # pass

def joint_infos_to_numpy(joint_infos):
  joint_infos_np = []
  for part_joint_info in joint_infos:
    for zz in ["dir", "center"]:
      # if isinstance(part_joint_info["axis"][zz], np.array):
      part_joint_info["axis"][zz] = part_joint_info["axis"][zz].detach().numpy()
    joint_infos_np.append(part_joint_info)
  return joint_infos_np


def normalie_pc_bbox_batched(pc, rt_stats=False):
  pc_min = torch.min(pc, dim=1, keepdim=True)[0]
  pc_max = torch.max(pc, dim=1, keepdim=True)[0]
  pc_center = 0.5 * (pc_min + pc_max)
  
  pc = pc - pc_center
  extents = pc_max - pc_min
  scale = torch.sqrt(torch.sum(extents ** 2, dim=-1, keepdim=True))
  
  pc = pc / torch.clamp(scale, min=1e-6)
  if rt_stats:
    return pc, pc_center, scale
  else:
    return pc



def scale_vertices_to_target_scale(vertices, target_scale):
  # vertices: bsz x N x 3;
  # target_scale: bsz x 1
#   normalized_vertices = normalize_vertices_scale_torch(vertices)
  normalized_vertices = normalie_pc_bbox_batched(vertices)
  normalized_vertices = normalized_vertices * target_scale.unsqueeze(1)
  return normalized_vertices

def compute_normals_o3d(verts, faces): #### assume no batching... ####
  mesh = o3d.geometry.TriangleMesh()
  # o3d_mesh_b.vertices = verts_b
        # o3d_mesh_b.triangles = np.array(faces_b, dtype=np.long)
  mesh.vertices = verts.detach().cpu().numpy()
  mesh.triangles = faces.detach().cpu().numpy().astype(np.long)
  verts_normals = mesh.compute_vertex_normals(normalized=True)
  verts_normals = torch.from_numpy(verts_normals, dtype=torch.float32).cuda()
  return verts_normals
  
  
def get_vals_via_nearest_neighbours(pts_src, pts_tar, val_tar):
  ### n_src x 3 ---> n_src x n_tar
  dist_src_tar = torch.sum((pts_src.unsqueeze(-2) - pts_tar.unsqueeze(0)) ** 2, dim=-1)
  minn_dists_src_tar, minn_dists_tar_idxes = torch.min(dist_src_tar, dim=-1) ### n_src
  selected_src_val = batched_index_select(values=val_tar, indices=minn_dists_tar_idxes, dim=0) ### n_src x dim
  return selected_src_val
  
  

## sample conected componetn start from selected_verts
def sample_bfs_component(selected_vert, faces, max_num_grids):
  vert_idx_to_adj_verts = {}
  for i_f, cur_f in enumerate(faces):
    # for i0, v0 in enumerate(cur_f):
    for i0 in range(len(cur_f)):
      v0 = cur_f[i0] - 1
      i1 = (i0 + 1) % len(cur_f)
      v1 = cur_f[i1] - 1
      if v0 not in vert_idx_to_adj_verts:
        vert_idx_to_adj_verts[v0] = {v1: 1}
      else:
        vert_idx_to_adj_verts[v0][v1] = 1
      if v1 not in vert_idx_to_adj_verts:
        vert_idx_to_adj_verts[v1] = {v0: 1}
      else:
        vert_idx_to_adj_verts[v1][v0] = 1
  vert_idx_to_visited = {} # whether visisted here # 
  vis_que = [selected_vert]
  vert_idx_to_visited[selected_vert] = 1
  visited = [selected_vert]
  while len(vis_que) > 0 and len(visited) < max_num_grids:
    cur_frnt_vert = vis_que[0]
    vis_que.pop(0)
    if cur_frnt_vert in vert_idx_to_adj_verts:
      cur_frnt_vert_adjs = vert_idx_to_adj_verts[cur_frnt_vert]
      for adj_vert in cur_frnt_vert_adjs:
        if adj_vert not in vert_idx_to_visited:
          vert_idx_to_visited[adj_vert] = 1
          vis_que.append(adj_vert)
          visited.append(adj_vert)
  if len(visited) >= max_num_grids:
    visited = visited[: max_num_grids - 1]
  return visited

def select_faces_via_verts(selected_verts, faces):
  if not isinstance(selected_verts, list):
    selected_verts = selected_verts.tolist()
  # selected_verts_dict = {ii: 1 for ii in selected_verts}
  old_idx_to_new_idx = {v + 1: ii + 1 for ii, v in enumerate(selected_verts)} ####### v + 1: ii + 1 --> for the selected_verts
  new_faces = []
  for i_f, cur_f in enumerate(faces):
    cur_new_f = []
    valid = True
    for cur_v in cur_f:
      if cur_v in old_idx_to_new_idx:
        cur_new_f.append(old_idx_to_new_idx[cur_v])
      else:
        valid  = False
        break
    if valid:
      new_faces.append(cur_new_f)
  return new_faces 
      

def convert_grid_content_to_grid_pts(content_value, grid_size):
  flat_grid = torch.zeros([grid_size ** 3], dtype=torch.long)
  cur_idx = flat_grid.size(0) - 1
  while content_value > 0:
    flat_grid[cur_idx] = content_value % grid_size
    content_value = content_value // grid_size
    cur_idx -= 1
  grid_pts = flat_grid.contiguous().view(grid_size, grid_size, grid_size).contiguous()
  return grid_pts

# 0.2
def warp_coord(sampled_gradients, val, reflect=False): # val from [0.0, 1.0] # from the 0.0 
  # assume single value as inputs
  grad_values = sampled_gradients.tolist()
  # mid_val
  mid_val = grad_values[0] * 0.2 + grad_values[1] * 0.2 + grad_values[2] * 0.1
  if reflect:
    grad_values[3] = grad_values[1]
    grad_values[4] = grad_values[0]

  # if not reflect:
  accum_val = 0.0
  for i_val in range(len(grad_values)):
    if val > 0.2 * (i_val + 1) and i_val < 4: # if i_val == 4, directly use the reamining length * corresponding gradient value
      accum_val += grad_values[i_val] * 0.2
    else:
      accum_val += grad_values[i_val] * (val - 0.2 * i_val)
      break
  return accum_val # modified value

def random_shift(vertices, shift_factor=0.25):
  """Apply random shift to vertices."""
  # max_shift_pos = tf.cast(255 - tf.reduce_max(vertices, axis=0), tf.float32)
  
  # max_shift_pos = tf.maximum(max_shift_pos, 1e-9)

  # max_shift_neg = tf.cast(tf.reduce_min(vertices, axis=0), tf.float32)
  # max_shift_neg = tf.maximum(max_shift_neg, 1e-9)

  # shift = tfd.TruncatedNormal(
  #     tf.zeros([1, 3]), shift_factor*255, -max_shift_neg,
  #     max_shift_pos).sample()
  # shift = tf.cast(shift, tf.int32)
  # vertices += shift

  minn_tensor = torch.tensor([1e-9], dtype=torch.float32)
  
  max_shift_pos = (255 - torch.max(vertices, dim=0)[0]).float()
  max_shift_pos = torch.maximum(max_shift_pos, minn_tensor)
  max_shift_neg = (torch.min(vertices, dim=0)[0]).float()
  max_shift_neg = torch.maximum(max_shift_neg, minn_tensor)
  
  shift = torch.zeros((1, 3), dtype=torch.float32)
  # torch.nn.init.trunc_normal_(shift, 0., shift_factor * 255., -max_shift_neg, max_shift_pos)
  for i_s in range(shift.size(-1)):
    cur_axis_max_shift_neg = max_shift_neg[i_s].item()
    cur_axis_max_shift_pos = max_shift_pos[i_s].item()
    cur_axis_shift = torch.zeros((1,), dtype=torch.float32)

    torch.nn.init.trunc_normal_(cur_axis_shift, 0., shift_factor * 255., -cur_axis_max_shift_neg, cur_axis_max_shift_pos)
    shift[:, i_s] = cur_axis_shift.item()
    
  shift = shift.long()
  vertices += shift

  return vertices

def safe_transpose(tsr, dima, dimb):
  tsr = tsr.contiguous().transpose(dima, dimb).contiguous()
  return tsr

def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)

def read_obj_file_ours(obj_fn, sub_one=False):
  vertices = []
  faces = []
  with open(obj_fn, "r") as rf:
    for line in rf:
      items = line.strip().split(" ")
      if items[0] == 'v':
        cur_verts = items[1:]
        cur_verts = [float(vv) for vv in cur_verts]
        vertices.append(cur_verts)
      elif items[0] == 'f':
        cur_faces = items[1:] # faces
        cur_face_idxes = []
        for cur_f in cur_faces:
          try:
            cur_f_idx = int(cur_f.split("/")[0])
          except:
            cur_f_idx = int(cur_f.split("//")[0])
          cur_face_idxes.append(cur_f_idx if not sub_one else cur_f_idx - 1)
        faces.append(cur_face_idxes)
    rf.close()
  vertices = np.array(vertices, dtype=np.float)
  return vertices, faces

def read_obj_file(obj_file):
  """Read vertices and faces from already opened file."""
  vertex_list = []
  flat_vertices_list = []
  flat_vertices_indices = {}
  flat_triangles = []

  for line in obj_file:
    tokens = line.split()
    if not tokens:
      continue
    line_type = tokens[0]
    # We skip lines not starting with v or f.
    if line_type == 'v': # 
      vertex_list.append([float(x) for x in tokens[1:]])
    elif line_type == 'f':
      triangle = []
      for i in range(len(tokens) - 1):
        vertex_name = tokens[i + 1]
        if vertex_name in flat_vertices_indices: # triangles
          triangle.append(flat_vertices_indices[vertex_name])
          continue
        flat_vertex = []
        for index in six.ensure_str(vertex_name).split('/'):
          if not index:
            continue
          # obj triangle indices are 1 indexed, so subtract 1 here.
          flat_vertex += vertex_list[int(index) - 1]
        flat_vertex_index = len(flat_vertices_list)
        flat_vertices_list.append(flat_vertex)
        # flat_vertex_index
        flat_vertices_indices[vertex_name] = flat_vertex_index
        triangle.append(flat_vertex_index)
      flat_triangles.append(triangle)

  return np.array(flat_vertices_list, dtype=np.float32), flat_triangles


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def safe_transpose(x, dim1, dim2):
    x = x.contiguous().transpose(dim1, dim2).contiguous()
    return x

def merge_meshes(vertices_list, faces_list):
  tot_vertices = []
  tot_faces = []
  nn_verts = 0
  for cur_vertices, cur_faces in zip(vertices_list, faces_list):
    tot_vertices.append(cur_vertices)
    new_cur_faces = []
    for cur_face_idx in cur_faces:
      new_cur_face_idx = [vert_idx + nn_verts for vert_idx in cur_face_idx]
      new_cur_faces.append(new_cur_face_idx)
    nn_verts += cur_vertices.shape[0]
    tot_faces += new_cur_faces # get total-faces
  tot_vertices = np.concatenate(tot_vertices, axis=0)
  return tot_vertices, tot_faces


def read_obj(obj_path):
  """Open .obj file from the path provided and read vertices and faces."""

  with open(obj_path) as obj_file:
    return read_obj_file_ours(obj_path, sub_one=True)
    # return read_obj_file(obj_file)




def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2)) # normalize the diagonal line to 1.
  return vertices / scale


def get_vertices_center(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vert_center

def get_batched_vertices_center(vertices):
  vert_min = vertices.min(axis=1)
  vert_max = vertices.max(axis=1)
  vert_center = 0.5 * (vert_min + vert_max)
  return vert_center

def get_vertices_scale(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2))
  return scale

def quantize_verts(verts, n_bits=8, min_range=None, max_range=None):
  """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
  min_range = -0.5 if min_range is None else min_range
  max_range = 0.5 if max_range is None else max_range
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.astype('int32')

def quantize_verts_torch(verts, n_bits=8, min_range=None, max_range=None):
  min_range = -0.5 if min_range is None else min_range
  max_range = 0.5 if max_range is None else max_range
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.long()

def dequantize_verts(verts, n_bits=8, add_noise=False, min_range=None, max_range=None):
  """Convert quantized vertices to floats."""
  min_range = -0.5 if min_range is None else min_range
  max_range = 0.5 if max_range is None else max_range
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts

def dequantize_verts_torch(verts, n_bits=8, add_noise=False, min_range=None, max_range=None):
  min_range = -0.5 if min_range is None else min_range
  max_range = 0.5 if max_range is None else max_range
  range_quantize = 2**n_bits - 1
  verts = verts.float()
  verts = verts * (max_range - min_range) / range_quantize + min_range
  # if add_noise:
  #   verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts


### dump vertices and faces to the obj file
def write_obj(vertices, faces, file_path, transpose=True, scale=1.):
  """Write vertices and faces to obj."""
  if transpose:
    vertices = vertices[:, [1, 2, 0]]
  vertices *= scale
  if faces is not None:
    if min(min(faces)) == 0:
      f_add = 1
    else:
      f_add = 0
  else:
    faces = []
  with open(file_path, 'w') as f:
    for v in vertices:
      f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for face in faces:
      line = 'f'
      for i in face:
        line += ' {}'.format(i + f_add)
      line += '\n'
      f.write(line)


def face_to_cycles(face):
  """Find cycles in face."""
  g = nx.Graph()
  for v in range(len(face) - 1):
    g.add_edge(face[v], face[v + 1])
  g.add_edge(face[-1], face[0])
  return list(nx.cycle_basis(g))


def flatten_faces(faces):
  """Converts from list of faces to flat face array with stopping indices."""
  if not faces:
    return np.array([0])
  else:
    l = [f + [-1] for f in faces[:-1]]
    l += [faces[-1] + [-2]]
    return np.array([item for sublist in l for item in sublist]) + 2  # pylint: disable=g-complex-comprehension


def unflatten_faces(flat_faces):
  """Converts from flat face sequence to a list of separate faces."""
  def group(seq):
    g = []
    for el in seq:
      if el == 0 or el == -1:
        yield g
        g = []
      else:
        g.append(el - 1)
    yield g
  outputs = list(group(flat_faces - 1))[:-1]
  # Remove empty faces
  return [o for o in outputs if len(o) > 2]



def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8, remove_du=True):
  """Quantize vertices, remove resulting duplicates and reindex faces."""
  vertices = quantize_verts(vertices, quantization_bits)
  vertices, inv = np.unique(vertices, axis=0, return_inverse=True) # return inverse and unique the vertices

  # 
  if opt.dataset.sort_dist:
    if opt.model.debug:
      print("sorting via dist...")
    vertices_max = np.max(vertices, axis=0)
    vertices_min = np.min(vertices, axis=0)
    dist_vertices = np.minimum(np.abs(vertices - np.array([[vertices_min[0], vertices_min[1], 0]])), np.abs(vertices - np.array([[vertices_max[0], vertices_max[1], 0]])))
    dist_vertices = np.concatenate([dist_vertices[:, 0:1] + dist_vertices[:, 1:2], dist_vertices[:, 2:]], axis=-1)
    sort_inds = np.lexsort(dist_vertices.T)
  else:
    # Sort vertices by z then y then x.
    sort_inds = np.lexsort(vertices.T) # sorted indices...
  vertices = vertices[sort_inds]

  # Re-index faces and tris to re-ordered vertices.
  faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
  if tris is not None:
    tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

  # Merging duplicate vertices and re-indexing the faces causes some faces to
  # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
  # sub-faces.
  sub_faces = []
  for f in faces:
    cliques = face_to_cycles(f)
    for c in cliques:
      c_length = len(c)
      # Only append faces with more than two verts.
      if c_length > 2:
        d = np.argmin(c)
        # Cyclically permute faces just that first index is the smallest.
        sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
  faces = sub_faces
  if tris is not None:
    tris = np.array([v for v in tris if len(set(v)) == len(v)])

  # Sort faces by lowest vertex indices. If two faces have the same lowest
  # index then sort by next lowest and so on.
  faces.sort(key=lambda f: tuple(sorted(f)))
  if tris is not None:
    tris = tris.tolist()
    tris.sort(key=lambda f: tuple(sorted(f)))
    tris = np.array(tris)

  # After removing degenerate faces some vertices are now unreferenced. # Vertices
  # Remove these. # Vertices
  num_verts = vertices.shape[0]
  # print(f"remove_du: {remove_du}")
  if remove_du: ##### num_verts
    print("Removing du..")
    try:
      vert_connected = np.equal(
          np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
      vertices = vertices[vert_connected]
    

      # Re-index faces and tris to re-ordered vertices.
      vert_indices = (
          np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
      faces = [vert_indices[f].tolist() for f in faces]
    except:
      pass
  if tris is not None:
    tris = np.array([vert_indices[t].tolist() for t in tris])

  return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8, recenter_mesh=True, remove_du=True):
  """Process mesh vertices and faces."""

  

  # Transpose so that z-axis is vertical.
  vertices = vertices[:, [2, 0, 1]]

  if recenter_mesh:
    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

  # Quantize and sort vertices, remove resulting duplicates, sort and reindex
  # faces.
  vertices, faces, _ = quantize_process_mesh(
      vertices, faces, quantization_bits=quantization_bits, remove_du=remove_du) ##### quantize_process_mesh
  
  # unflatten_faces = np.array(faces, dtype=np.long) ### start from zero

  # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
  faces = flatten_faces(faces)

  # Discard degenerate meshes without faces.
  return {
      'vertices': vertices,
      'faces': faces,
  }


def process_mesh_ours(vertices, faces, quantization_bits=8, recenter_mesh=True, remove_du=True):
  """Process mesh vertices and faces."""
  # Transpose so that z-axis is vertical.
  vertices = vertices[:, [2, 0, 1]]

  if recenter_mesh:
    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

  # Quantize and sort vertices, remove resulting duplicates, sort and reindex
  # faces.
  quant_vertices, faces, _ = quantize_process_mesh(
      vertices, faces, quantization_bits=quantization_bits, remove_du=remove_du) ##### quantize_process_mesh
  vertices = dequantize_verts(quant_vertices) #### dequantize vertices ####
  ### vertices: nn_verts x 3
  # try:
  #   # print("faces", faces)
  #   unflatten_faces = np.array(faces, dtype=np.long)
  # except:
  #   print("faces", faces)
  #   raise ValueError("Something bad happened when processing meshes...")
  
  # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.

  faces = flatten_faces(faces)

  # Discard degenerate meshes without faces.
  return {
      'vertices': quant_vertices,
      'faces': faces,
      # 'unflatten_faces': unflatten_faces,
      'dequant_vertices': vertices,
      'class_label': 0
  }

def read_mesh_from_obj_file(fn, quantization_bits=8, recenter_mesh=True, remove_du=True):
  vertices, faces = read_obj(fn)
  # print(vertices.shape)
  mesh_dict = process_mesh_ours(vertices, faces, quantization_bits=quantization_bits, recenter_mesh=recenter_mesh, remove_du=remove_du)
  return mesh_dict

def process_mesh_list(vertices, faces, quantization_bits=8, recenter_mesh=True):
  """Process mesh vertices and faces."""

  vertices = [cur_vert[:, [2, 0, 1]] for cur_vert in vertices]

  tot_vertices = np.concatenate(vertices, axis=0) # center and scale of those vertices
  vert_center = get_vertices_center(tot_vertices)
  vert_scale = get_vertices_scale(tot_vertices)

  processed_vertices, processed_faces = [], []

  for cur_verts, cur_faces in zip(vertices, faces):
    # print(f"current vertices: {cur_verts.shape}, faces: {len(cur_faces)}")
    normalized_verts = (cur_verts - vert_center) / vert_scale
    cur_processed_verts, cur_processed_faces, _ = quantize_process_mesh(
      normalized_verts, cur_faces, quantization_bits=quantization_bits
    )
    processed_vertices.append(cur_processed_verts)
    processed_faces.append(cur_processed_faces)
  vertices, faces = merge_meshes(processed_vertices, processed_faces)
  faces = flatten_faces(faces=faces)


  # Discard degenerate meshes without faces.
  return {
      'vertices': vertices,
      
      'faces': faces,
      
  }


def plot_sampled_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample['left'], v_sample['rgt']]
  part_face_samples = [f_sample['left'], f_sample['rgt']]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  if predict_joint:
    pred_dir = v_sample['joint_dir']
    pred_pvp = v_sample['joint_pvp']
    print("pred_dir", pred_dir.shape, pred_dir)
    print("pred_pvp", pred_pvp.shape, pred_pvp)
  else:
    pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  
  


  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)
      # and write this obj file?
      # write_obj(vertices, faces, file_path, transpose=True, scale=1.):
      # write mesh objs
      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
        
      

  ###### plot mesh (predicted) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}.obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
  ###### plot mesh (predicted) ######


  ###### plot mesh (translated) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      cur_s_pred_pvp = pred_pvp[i_s]
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]

          if i_p == 1:
            # min_rngs = cur_s_cur_part_vertices.min(1)
            # max_rngs = cur_s_cur_part_vertices.max(1)
            min_rngs = cur_s_cur_part_vertices.min(0)
            max_rngs = cur_s_cur_part_vertices.max(0)
            # shifted; cur_s_pred_pvp
            # shifted = np.array([0., cur_s_pred_pvp[1] - max_rngs[1], cur_s_pred_pvp[2] - min_rngs[2]], dtype=np.float)
            # shifted = np.reshape(shifted, [1, 3]) # 
            cur_s_pred_pvp = np.array([0., max_rngs[1], min_rngs[2]], dtype=np.float32)
            pvp_sample_pred_err = np.sum((cur_s_pred_pvp - pred_pvp[i_s]) ** 2)
            # print prediction err, pred pvp and real pvp
            # print("cur_s, sample_pred_pvp_err:", pvp_sample_pred_err.item(), ";real val:", cur_s_pred_pvp, "; pred_val:", pred_pvp[i_s])
            pred_pvp[i_s] = cur_s_pred_pvp
            shifted = np.zeros((1, 3), dtype=np.float32)
            cur_s_cur_part_vertices = cur_s_cur_part_vertices + shifted # shift vertices... # min_rngs
          # shifted
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_shifted.obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
  ###### plot mesh (translated) ######



  ###### plot mesh (rotated) ######
  if predict_joint:
    from revolute_transform import revoluteTransform
    tot_samples_mesh_dict = []
    for i_s in range(tot_n_samples):
        cur_s_tot_vertices = []
        cur_s_tot_faces = []
        cur_s_n_vertices = 0
      
        # cur_s_pred_dir = pred_dir[i_s]
        cur_s_pred_pvp = pred_pvp[i_s]
        print("current pred dir:", cur_s_pred_dir, "; current pred pvp:", cur_s_pred_pvp)
        cur_s_pred_dir = np.array([1.0, 0.0, 0.0], dtype=np.float)
        # cur_s_pred_pvp = cur_s_pred_pvp[[1, 2, 0]]

        for i_p in range(tot_n_part):
            cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
            cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                            cur_s_cur_part_mesh_dict['faces']
                      
            if i_p == 1:
              cur_s_cur_part_vertices, _ = revoluteTransform(cur_s_cur_part_vertices, cur_s_pred_pvp, cur_s_pred_dir, 0.5 * np.pi) # reverse revolute vertices of the upper piece
              cur_s_cur_part_vertices = cur_s_cur_part_vertices[:, :3] # 
            cur_s_cur_part_new_faces = []
            for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
                cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
                cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
            cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
            cur_s_tot_vertices.append(cur_s_cur_part_vertices)
            # print(f"i_s: {i_s}, i_p: {i_p}, n_vertices: {cur_s_cur_part_vertices.shape[0]}")
            cur_s_tot_faces += cur_s_cur_part_new_faces

        cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
        # print(f"i_s: {i_s}, n_cur_s_tot_vertices: {cur_s_tot_vertices.shape[0]}")
        cur_s_mesh_dict = {
            'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
        }
        tot_samples_mesh_dict.append(cur_s_mesh_dict)
    # plot_meshes(tot_samples_mesh_dict, ax_lims=0.5, mesh_sv_fn=f"./figs/training_step_{n}_part_{tot_n_part}_rot.png")  # plot the mesh;
    for i_s in range(tot_n_samples):
      cur_mesh = tot_samples_mesh_dict[i_s]
      cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
      # rotated mesh fn
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_rot.obj")
      # write object in the file...
      if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
        write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)




def sample_pts_from_mesh(vertices, faces, npoints=512, minus_one=True):
    return vertices
    sampled_pcts = []
    pts_to_seg_idx = []
    # triangles and pts
    minus_val = 0 if not minus_one else 1
    for i in range(len(faces)): # 
        cur_face = faces[i]
        n_tris = len(cur_face) - 2
        v_as, v_bs, v_cs = [cur_face[0] for _ in range(n_tris)], cur_face[1: len(cur_face) - 1], cur_face[2: len(cur_face)]
        for v_a, v_b, v_c in zip(v_as, v_bs, v_cs):
          
            v_a, v_b, v_c = vertices[v_a - minus_val], vertices[v_b - minus_val], vertices[v_c - minus_val]
            ab, ac = v_b - v_a, v_c - v_a
            cos_ab_ac = (np.sum(ab * ac) / np.clip(np.sqrt(np.sum(ab ** 2)) * np.sqrt(np.sum(ac ** 2)), a_min=1e-9,
                                                  a_max=9999999.)).item()
            sin_ab_ac = math.sqrt(min(max(0., 1. - cos_ab_ac ** 2), 1.))
            cur_area = 0.5 * sin_ab_ac * np.sqrt(np.sum(ab ** 2)).item() * np.sqrt(np.sum(ac ** 2)).item()

            cur_sampled_pts = int(cur_area * npoints)
            cur_sampled_pts = 1 if cur_sampled_pts == 0 else cur_sampled_pts
            # if cur_sampled_pts == 0:

            tmp_x, tmp_y = np.random.uniform(0, 1., (cur_sampled_pts,)).tolist(), np.random.uniform(0., 1., (
            cur_sampled_pts,)).tolist()

            for xx, yy in zip(tmp_x, tmp_y):
                sqrt_xx, sqrt_yy = math.sqrt(xx), math.sqrt(yy)
                aa = 1. - sqrt_xx
                bb = sqrt_xx * (1. - yy)
                cc = yy * sqrt_xx
                cur_pos = v_a * aa + v_b * bb + v_c * cc
                sampled_pcts.append(cur_pos)
                # pts_to_seg_idx.append(cur_tri_seg)

    # seg_idx_to_sampled_pts = {}
    sampled_pcts = np.array(sampled_pcts, dtype=np.float)

    return sampled_pcts


def fps_fr_numpy(np_pts, n_sampling=4096):
  pts = torch.from_numpy(np_pts).float().cuda()
  pts_fps_idx = farthest_point_sampling(pts.unsqueeze(0), n_sampling=n_sampling) # farthes points sampling ##
  pts = pts[pts_fps_idx].cpu()
  return pts


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx


def plot_sampled_meshes_single_part(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  # not predict joints here
  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  # else:
  #   pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)

      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
        

def plot_sampled_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  # tot_n_part = 2

  # not predict joints here
  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  # else:
  #   pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)

      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
         

def filter_masked_vertices(vertices, mask_indicator):
  # vertices: n_verts x 3 
  mask_indicator = np.reshape(mask_indicator, (vertices.shape[0], 3))
  tot_vertices = []
  for i_v in range(vertices.shape[0]):
    cur_vert = vertices[i_v]
    cur_vert_indicator = mask_indicator[i_v][0].item()
    if cur_vert_indicator < 0.5:
      tot_vertices.append(cur_vert)
  tot_vertices = np.array(tot_vertices)
  return tot_vertices


def plot_sampled_ar_subd_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=0, ):
  if not os.path.exists(sv_mesh_folder): ### vertices_mask
    os.mkdir(sv_mesh_folder)
  ### v_sample: bsz x nn_verts x 3
  vertices_mask = v_sample['vertices_mask']
  vertices = v_sample['vertices']
  faces = f_sample['faces']
  num_face_indices = f_sample['num_face_indices'] #### num_faces_indices
  bsz = vertices.shape[0]
  
  for i_bsz in range(bsz):
    cur_vertices = vertices[i_bsz]
    cur_vertices_mask = vertices_mask[i_bsz]
    cur_faces = faces[i_bsz]
    cur_num_face_indices = num_face_indices[i_bsz]
    cur_nn_verts = cur_vertices_mask.sum(-1).item()
    cur_nn_verts = int(cur_nn_verts)
    cur_vertices = cur_vertices[:cur_nn_verts]
    cur_faces = unflatten_faces(
                      cur_faces[:int(cur_num_face_indices)])

    cur_num_faces = len(cur_faces)
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_inst_{i_bsz}.obj")
    # cur_context_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}_context.obj")
    print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_nn_verts}, nn_faces: {cur_num_faces}")
    if cur_nn_verts > 0 and cur_num_faces > 0:
      write_obj(cur_vertices, cur_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
    
  

def plot_sampled_meshes_single_part_for_pretraining(v_sample, f_sample, context, sv_mesh_folder, cur_step=0, predict_joint=True, with_context=True):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]

  context_vertices = [context['vertices']]
  context_faces = [context['faces']]
  context_vertices_mask = [context['vertices_mask']]
  context_faces_mask = [context['faces_mask']]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  # not predict joints here
  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  # else:
  #   pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)

  # 
  

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      context_mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )

          cur_context_vertices = context_vertices[i_p][i_n]
          cur_context_faces = context_faces[i_p][i_n]
          cur_context_vertices_mask = context_vertices_mask[i_p][i_n]
          cur_context_faces_mask = context_faces_mask[i_p][i_n]
          cur_nn_vertices = np.sum(cur_context_vertices_mask).item()
          cur_nn_faces = np.sum(cur_context_faces_mask).item()
          cur_nn_vertices, cur_nn_faces = int(cur_nn_vertices), int(cur_nn_faces)
          cur_context_vertices  = cur_context_vertices[:cur_nn_vertices]
          if 'mask_vertices_flat_indicator' in context:
            cur_context_vertices_mask_indicator = context['mask_vertices_flat_indicator'][i_n]
            cur_context_vertices_mask_indicator = cur_context_vertices_mask_indicator[:cur_nn_vertices * 3]
            cur_context_vertices = filter_masked_vertices(cur_context_vertices, cur_context_vertices_mask_indicator)
          cur_context_faces = cur_context_faces[:cur_nn_faces] # context faces...
          context_mesh_dict = {
            'vertices': dequantize_verts(cur_context_vertices, n_bits=8), 'faces': unflatten_faces(cur_context_faces)
          }
          context_mesh_list.append(context_mesh_dict)

      tot_mesh_list.append(mesh_list)

      # if with_context:
      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_context_mesh = context_mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        cur_context_vertices, cur_context_faces = cur_context_mesh['vertices'], cur_context_mesh['faces']
        # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_context_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}_context.obj")
        print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
        if cur_context_vertices.shape[0] > 0 and len(cur_context_faces) > 0:
          write_obj(cur_context_vertices, cur_context_faces, cur_context_mesh_sv_fn, transpose=True, scale=1.)


def plot_grids_for_pretraining_ml(v_sample, sv_mesh_folder="", cur_step=0, context=None):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  mesh_list = []
  context_mesh_list = []
  tot_n_samples = v_sample['vertices'].shape[0]

  for i_n in range(tot_n_samples):
      mesh_list.append(
          {
              'vertices': v_sample['vertices'][i_n][:v_sample['num_vertices'][i_n]],
              'faces': []
          }
      )

      cur_context_vertices = context['vertices'][i_n]
      cur_context_vertices_mask = context['vertices_mask'][i_n]
      cur_nn_vertices = np.sum(cur_context_vertices_mask).item()
      cur_nn_vertices = int(cur_nn_vertices)
      cur_context_vertices  = cur_context_vertices[:cur_nn_vertices]
      if 'mask_vertices_flat_indicator' in context:
        cur_context_vertices_mask_indicator = context['mask_vertices_flat_indicator'][i_n]
        cur_context_vertices_mask_indicator = cur_context_vertices_mask_indicator[:cur_nn_vertices * 3]
        cur_context_vertices = filter_masked_vertices(cur_context_vertices, cur_context_vertices_mask_indicator)
      context_mesh_dict = {
        'vertices': dequantize_verts(cur_context_vertices, n_bits=8), 'faces': []
      }
      context_mesh_list.append(context_mesh_dict)

  # tot_mesh_list.append(mesh_list)

  # if with_context:
  for i_n in range(tot_n_samples):
    cur_mesh = mesh_list[i_n]
    cur_context_mesh = context_mesh_list[i_n]
    cur_mesh_vertices = cur_mesh['vertices']
    cur_context_vertices = cur_context_mesh['vertices']
    # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_n}.obj")
    cur_context_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_n}_context.obj")
    # print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
    print(f"saving the sample to {cur_mesh_sv_fn}, context sample to {cur_context_mesh_sv_fn}")
    if cur_mesh_vertices.shape[0] > 0:
      write_obj(cur_mesh_vertices, None, cur_mesh_sv_fn, transpose=True, scale=1.)
    if cur_context_vertices.shape[0] > 0:
      write_obj(cur_context_vertices, None, cur_context_mesh_sv_fn, transpose=True, scale=1.)
  

def get_grid_content_from_grids(grid_xyzs, grid_values, grid_size=2):
  cur_bsz_grid_xyzs = grid_xyzs # grid_length x 3 # grids pts for a sinlge batch
  cur_bsz_grid_values = grid_values  # grid_length x gs x gs x gs
  pts = []
  for i_grid in range(cur_bsz_grid_xyzs.shape[0]): # cur_bsz_grid_xyzs
    cur_grid_xyz = cur_bsz_grid_xyzs[i_grid].tolist()
    if cur_grid_xyz[0] == -1 or cur_grid_xyz[1] == -1 or cur_grid_xyz[2] == -1:
      break
    if len(cur_bsz_grid_values.shape) > 1: 
      cur_grid_values = cur_bsz_grid_values[i_grid]
    else:
      cur_grid_content = cur_bsz_grid_values[i_grid].item()
      if cur_grid_content >= MASK_GRID_VALIE:
        continue
      inde = 2
      cur_grid_values = []
      for i_s in range(grid_size ** 3):
        cur_mod_value = cur_grid_content % inde
        cur_grid_content = cur_grid_content // inde
        cur_grid_values = [cur_mod_value] + cur_grid_values # higher values should be put to the front of the list
      cur_grid_values = np.array(cur_grid_values, dtype=np.long)
      cur_grid_values = np.reshape(cur_grid_values, (grid_size, grid_size, grid_size))
  # if words 
  # flip words 
    for i_x in range(cur_grid_values.shape[0]):
      for i_y in range(cur_grid_values.shape[1]):
        for i_z in range(cur_grid_values.shape[2]):
          cur_grid_xyz_value = int(cur_grid_values[i_x, i_y, i_z].item())
          if cur_grid_xyz_value > 0.5:
            cur_x, cur_y, cur_z = cur_grid_xyz[0] * grid_size + i_x, cur_grid_xyz[1] * grid_size + i_y, cur_grid_xyz[2] * grid_size + i_z
            pts.append([cur_x, cur_y, cur_z])
  return pts

def plot_grids_for_pretraining(v_sample, sv_mesh_folder="", cur_step=0, context=None, sv_mesh_fn=None):
  
  ##### plot grids
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  # part_vertex_samples = [v_sample] # vertex samples
  # part_face_samples = [f_sample] # face samples

  grid_xyzs = v_sample['grid_xyzs']
  grid_values = v_sample['grid_values']

  bsz = grid_xyzs.shape[0]
  grid_size = opt.vertex_model.grid_size
  

  for i_bsz in range(bsz):
    cur_bsz_grid_xyzs = grid_xyzs[i_bsz] # grid_length x 3
    cur_bsz_grid_values = grid_values[i_bsz] # grid_length x gs x gs x gs
    pts = []
    for i_grid in range(cur_bsz_grid_xyzs.shape[0]): # cur_bsz_grid_xyzs
      cur_grid_xyz = cur_bsz_grid_xyzs[i_grid].tolist()
      if cur_grid_xyz[0] == -1 or cur_grid_xyz[1] == -1 or cur_grid_xyz[2] == -1:
        break
      if len(cur_bsz_grid_values.shape) > 1: 
        cur_grid_values = cur_bsz_grid_values[i_grid]
      else:
        cur_grid_content = cur_bsz_grid_values[i_grid].item()
        if cur_grid_content >= MASK_GRID_VALIE:
          continue
        inde = 2
        cur_grid_values = []
        for i_s in range(grid_size ** 3):
          cur_mod_value = cur_grid_content % inde
          cur_grid_content = cur_grid_content // inde
          cur_grid_values = [cur_mod_value] + cur_grid_values # higher values should be put to the front of the list
        cur_grid_values = np.array(cur_grid_values, dtype=np.long)
        cur_grid_values = np.reshape(cur_grid_values, (grid_size, grid_size, grid_size))
    # if 
      for i_x in range(cur_grid_values.shape[0]):
        for i_y in range(cur_grid_values.shape[1]):
          for i_z in range(cur_grid_values.shape[2]):
            cur_grid_xyz_value = int(cur_grid_values[i_x, i_y, i_z].item())
            if cur_grid_xyz_value > 0.5:
              cur_x, cur_y, cur_z = cur_grid_xyz[0] * grid_size + i_x, cur_grid_xyz[1] * grid_size + i_y, cur_grid_xyz[2] * grid_size + i_z
              pts.append([cur_x, cur_y, cur_z])
        
    
    if len(pts) == 0:
      print("zzz, len(pts) == 0")
      continue
    pts = np.array(pts, dtype=np.float32)
    # pts = center_vertices(pts)
    # pts = normalize_vertices_scale(pts)
    pts = pts[:, [2, 1, 0]]
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}.obj" if sv_mesh_fn is None else sv_mesh_fn)
    
    print(f"Saving obj to {cur_mesh_sv_fn}")
    write_obj(pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)


def plot_grids_for_pretraining_obj_corpus(v_sample, sv_mesh_folder="", cur_step=0, context=None, sv_mesh_fn=None):
  
  ##### plot grids
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  # part_vertex_samples = [v_sample] # vertex samples
  # part_face_samples = [f_sample] # face samples

  grid_xyzs = v_sample['grid_xyzs']
  grid_values = v_sample['grid_values']

  bsz = grid_xyzs.shape[0]
  grid_size = opt.vertex_model.grid_size
  

  for i_bsz in range(bsz):
    cur_bsz_grid_xyzs = grid_xyzs[i_bsz] # grid_length x 3
    cur_bsz_grid_values = grid_values[i_bsz] # grid_length x gs x gs x gs
    part_pts = []
    pts = []
    for i_grid in range(cur_bsz_grid_xyzs.shape[0]): # cur_bsz_grid_xyzs
      cur_grid_xyz = cur_bsz_grid_xyzs[i_grid].tolist()
      ##### grid_xyz; grid_
      if cur_grid_xyz[0] == -1 and cur_grid_xyz[1] == -1 and cur_grid_xyz[2] == -1:
        part_pts.append(pts)
        pts = []
        continue
      ##### cur_grid_xyz... #####
      elif not (cur_grid_xyz[0] >= 0 and cur_grid_xyz[1] >= 0 and cur_grid_xyz[2] >= 0):
        continue
      if len(cur_bsz_grid_values.shape) > 1: 
        cur_grid_values = cur_bsz_grid_values[i_grid]
      else:
        cur_grid_content = cur_bsz_grid_values[i_grid].item()
        if cur_grid_content >= MASK_GRID_VALIE: # mask grid value
          continue
        inde = 2
        cur_grid_values = []
        for i_s in range(grid_size ** 3):
          cur_mod_value = cur_grid_content % inde
          cur_grid_content = cur_grid_content // inde
          cur_grid_values = [cur_mod_value] + cur_grid_values # higher values should be put to the front of the list
        cur_grid_values = np.array(cur_grid_values, dtype=np.long)
        cur_grid_values = np.reshape(cur_grid_values, (grid_size, grid_size, grid_size))
    # if 
      for i_x in range(cur_grid_values.shape[0]):
        for i_y in range(cur_grid_values.shape[1]):
          for i_z in range(cur_grid_values.shape[2]):
            cur_grid_xyz_value = int(cur_grid_values[i_x, i_y, i_z].item())
            ##### gird-xyz-values #####
            if cur_grid_xyz_value > 0.5: # cur_grid_xyz_value
              cur_x, cur_y, cur_z = cur_grid_xyz[0] * grid_size + i_x, cur_grid_xyz[1] * grid_size + i_y, cur_grid_xyz[2] * grid_size + i_z
              pts.append([cur_x, cur_y, cur_z])
      
    if len(pts) > 0:
      part_pts.append(pts)
      pts = []
    tot_nn_pts = sum([len(aa) for aa in part_pts])
    if tot_nn_pts == 0:
      print("zzz, tot_nn_pts == 0")
      continue

    for i_p, pts in enumerate(part_pts):
      if len(pts) == 0:
        continue
      pts = np.array(pts, dtype=np.float32)
      pts = center_vertices(pts)
      # pts = normalize_vertices_scale(pts)
      pts = pts[:, [2, 1, 0]]
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}_ip_{i_p}.obj" if sv_mesh_fn is None else sv_mesh_fn)
      
      print(f"Saving {i_p}-th part obj to {cur_mesh_sv_fn}")
      write_obj(pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)



def plot_grids_for_pretraining_obj_part(v_sample, sv_mesh_folder="", cur_step=0, context=None, sv_mesh_fn=None):
  
  ##### plot grids
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  # part_vertex_samples = [v_sample] # vertex samples
  # part_face_samples = [f_sample] # face samples

  grid_xyzs = v_sample['grid_xyzs']
  grid_values = v_sample['grid_values']

  bsz = grid_xyzs.shape[0]
  grid_size = opt.vertex_model.grid_size
  

  for i_bsz in range(bsz):
    cur_bsz_grid_xyzs = grid_xyzs[i_bsz] # grid_length x 3
    cur_bsz_grid_values = grid_values[i_bsz] # grid_length x gs x gs x gs
    part_pts = []
    pts = []
    for i_grid in range(cur_bsz_grid_xyzs.shape[0]): # cur_bsz_grid_xyzs
      cur_grid_xyz = cur_bsz_grid_xyzs[i_grid].tolist()
      ##### grid_xyz; grid_
      if cur_grid_xyz[0] == -1 and cur_grid_xyz[1] == -1 and cur_grid_xyz[2] == -1:
        part_pts.append(pts)
        pts = []
        break
      elif cur_grid_xyz[0] == -1 and cur_grid_xyz[1] == -1 and cur_grid_xyz[2] == 0:
        part_pts.append(pts)
        pts = []
        continue
      ##### cur_grid_xyz... #####
      elif not (cur_grid_xyz[0] >= 0 and cur_grid_xyz[1] >= 0 and cur_grid_xyz[2] >= 0):
        continue
      if len(cur_bsz_grid_values.shape) > 1: 
        cur_grid_values = cur_bsz_grid_values[i_grid]
      else:
        cur_grid_content = cur_bsz_grid_values[i_grid].item()
        if cur_grid_content >= MASK_GRID_VALIE: # invalid jor dummy content value s
          continue
        inde = 2
        cur_grid_values = []
        for i_s in range(grid_size ** 3):
          cur_mod_value = cur_grid_content % inde
          cur_grid_content = cur_grid_content // inde
          cur_grid_values = [cur_mod_value] + cur_grid_values # higher values should be put to the front of the list
        cur_grid_values = np.array(cur_grid_values, dtype=np.long)
        cur_grid_values = np.reshape(cur_grid_values, (grid_size, grid_size, grid_size))
    # if 
      for i_x in range(cur_grid_values.shape[0]):
        for i_y in range(cur_grid_values.shape[1]):
          for i_z in range(cur_grid_values.shape[2]):
            cur_grid_xyz_value = int(cur_grid_values[i_x, i_y, i_z].item())
            ##### gird-xyz-values #####
            if cur_grid_xyz_value > 0.5: # cur_grid_xyz_value
              cur_x, cur_y, cur_z = cur_grid_xyz[0] * grid_size + i_x, cur_grid_xyz[1] * grid_size + i_y, cur_grid_xyz[2] * grid_size + i_z
              pts.append([cur_x, cur_y, cur_z])
      
    if len(pts) > 0:
      part_pts.append(pts)
      pts = []
    tot_nn_pts = sum([len(aa) for aa in part_pts])
    if tot_nn_pts == 0:
      print("zzz, tot_nn_pts == 0")
      continue

    for i_p, pts in enumerate(part_pts):
      if len(pts) == 0:
        continue
      pts = np.array(pts, dtype=np.float32)
      pts = center_vertices(pts)
      # pts = normalize_vertices_scale(pts)
      pts = pts[:, [2, 1, 0]]
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}_ip_{i_p}.obj" if sv_mesh_fn is None else sv_mesh_fn)
      
      print(f"Saving {i_p}-th part obj to {cur_mesh_sv_fn}")
      write_obj(pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)


def plot_grids_for_pretraining_ml(v_sample, sv_mesh_folder="", cur_step=0, context=None):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  # part_vertex_samples = [v_sample] # vertex samples
  # part_face_samples = [f_sample] # face samples

  grid_xyzs = v_sample['grid_xyzs']
  grid_values = v_sample['grid_values']

  context_grid_xyzs = context['grid_xyzs'] - 1
  # context_grid_values = context['grid_content']
  context_grid_values = context['mask_grid_content']

  bsz = grid_xyzs.shape[0]
  grid_size = opt.vertex_model.grid_size
  

  for i_bsz in range(bsz):
    cur_bsz_grid_pts = get_grid_content_from_grids(grid_xyzs[i_bsz], grid_values[i_bsz], grid_size=grid_size)
    cur_context_grid_pts = get_grid_content_from_grids(context_grid_xyzs[i_bsz], context_grid_values[i_bsz], grid_size=grid_size)

    if len(cur_bsz_grid_pts) > 0 and len(cur_context_grid_pts) > 0:
      cur_bsz_grid_pts = np.array(cur_bsz_grid_pts, dtype=np.float32)
      cur_bsz_grid_pts = center_vertices(cur_bsz_grid_pts)
      cur_bsz_grid_pts = normalize_vertices_scale(cur_bsz_grid_pts)
      cur_bsz_grid_pts = cur_bsz_grid_pts[:, [2, 1, 0]]
      #### plot current mesh / sampled points ####
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}.obj")
      print(f"Saving predicted grid points to {cur_mesh_sv_fn}")
      write_obj(cur_bsz_grid_pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)

      cur_context_grid_pts = np.array(cur_context_grid_pts, dtype=np.float32)
      cur_context_grid_pts = center_vertices(cur_context_grid_pts)
      cur_context_grid_pts = normalize_vertices_scale(cur_context_grid_pts)
      cur_context_grid_pts = cur_context_grid_pts[:, [2, 1, 0]]
      #### plot current mesh / sampled points ####
      cur_context_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}_context.obj")
      print(f"Saving context grid points to {cur_context_mesh_sv_fn}")
      write_obj(cur_context_grid_pts, None, cur_context_mesh_sv_fn, transpose=True, scale=1.)

    # print(f"Saving obj to {cur_mesh_sv_fn}")
    # write_obj(pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)

    # if len(cur_bsz_grid_pts) == 0:
    #   print("zzz, len(pts) == 0")
    #   continue
    # pts = np.array(pts, dtype=np.float32)
    # pts = center_vertices(pts)
    # pts = normalize_vertices_scale(pts)
    # pts = pts[:, [2, 1, 0]]
    # cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_bsz}.obj")
    
    # print(f"Saving obj to {cur_mesh_sv_fn}")
    # write_obj(pts, None, cur_mesh_sv_fn, transpose=True, scale=1.)



def plot_sampled_meshes_single_part_for_sampling(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  # not predict joints here
  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  # else:
  #   pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)

      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
        
     