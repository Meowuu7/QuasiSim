import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
import os

import trimesh
from pysdf import SDF

from uni_rep.rep_3d.dmtet import marching_tets_tetmesh, create_tetmesh_variables

def create_mt_variable(device):
    triangle_table = torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)

    num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=device)
    base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=device)
    v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=device))
    return triangle_table, num_triangles_table, base_tet_edges, v_id



def  extract_fields_from_tets(bound_min, bound_max, resolution, query_func, def_func=None):
    # load tet via resolution #
    # scale them via bounds #
    # extract the geometry #
    # /home/xueyi/gen/DeepMetaHandles/data/tets/100_compress.npz # strange #
    device = bound_min.device
    # if resolution in [64, 70, 80, 90, 100]:
    #     tet_fn = f"/home/xueyi/gen/DeepMetaHandles/data/tets/{resolution}_compress.npz"
    # else:
    tet_fn = f"/home/xueyi/gen/DeepMetaHandles/data/tets/{100}_compress.npz"
    tets = np.load(tet_fn)
    verts = torch.from_numpy(tets['vertices']).float().to(device) # verts positions 
    indices = torch.from_numpy(tets['tets']).long().to(device) # .to(self.device)
    # split #
    # verts; verts; #
    minn_verts, _ = torch.min(verts, dim=0)
    maxx_verts, _ = torch.max(verts, dim=0) # (3, ) # exporting the  
    # scale_verts = maxx_verts - minn_verts
    scale_bounds = bound_max - bound_min # scale bounds #
    
    ### scale the vertices ###
    scaled_verts = (verts - minn_verts.unsqueeze(0)) / (maxx_verts - minn_verts).unsqueeze(0) ### the maxx and minn verts scales ###
    
    # scaled_verts = (verts - minn_verts.unsqueeze(0)) / (maxx_verts - minn_verts).unsqueeze(0) ### the maxx and minn verts scales ###

    scaled_verts = scaled_verts * 2. - 1. # init the sdf filed viathe tet mesh vertices and the sdf values ##
    # scaled_verts = (scaled_verts * scale_bounds.unsqueeze(0)) + bound_min.unsqueeze(0) ## the scaled verts ###
    
    # scaled_verts = scaled_verts - scale_bounds.unsqueeze(0) / 2. # 
    # scaled_verts = scaled_verts -  bound_min.unsqueeze(0) - scale_bounds.unsqueeze(0) / 2.
    
    sdf_values = []
    N = 64
    query_bundles = N ** 3 ### N^3
    query_NNs = scaled_verts.size(0) // query_bundles
    if query_NNs * query_bundles < scaled_verts.size(0):
        query_NNs += 1
    for i_query in range(query_NNs):
        cur_bundle_st = i_query * query_bundles
        cur_bundle_ed = (i_query + 1) * query_bundles
        cur_bundle_ed = min(cur_bundle_ed, scaled_verts.size(0))
        cur_query_pts = scaled_verts[cur_bundle_st: cur_bundle_ed]
        if def_func is not None:
            cur_query_pts = def_func(cur_query_pts)
        cur_query_vals = query_func(cur_query_pts)
        sdf_values.append(cur_query_vals)
    sdf_values = torch.cat(sdf_values, dim=0)
    # print(f"queryed sdf values: {sdf_values.size()}") #
    
    GT_sdf_values = np.load("/home/xueyi/diffsim/DiffHand/assets/hand/100_sdf_values.npy", allow_pickle=True)
    GT_sdf_values = torch.from_numpy(GT_sdf_values).float().to(device)
    
    # intrinsic, tet values, pts values, sdf network #
    triangle_table, num_triangles_table, base_tet_edges, v_id = create_mt_variable(device)
    tet_table, num_tets_table = create_tetmesh_variables(device)
    
    sdf_values = sdf_values.squeeze(-1) # how the rendering # 
    
    # print(f"GT_sdf_values: {GT_sdf_values.size()}, sdf_values: {sdf_values.size()}, scaled_verts: {scaled_verts.size()}")
    # print(f"scaled_verts: {scaled_verts.size()}, ")
    # pos_nx3, sdf_n, tet_fx4, triangle_table, num_triangles_table, base_tet_edges, v_id,
    # return_tet_mesh=False, ori_v=None, num_tets_table=None, tet_table=None):
    # marching_tets_tetmesh ##
    verts, faces, tet_verts, tets = marching_tets_tetmesh(scaled_verts, sdf_values, indices, triangle_table, num_triangles_table, base_tet_edges, v_id, return_tet_mesh=True, ori_v=scaled_verts, num_tets_table=num_tets_table, tet_table=tet_table)
    ### use the GT sdf values for the marching tets ###
    GT_verts,  GT_faces,  GT_tet_verts,  GT_tets = marching_tets_tetmesh(scaled_verts, GT_sdf_values, indices, triangle_table, num_triangles_table, base_tet_edges, v_id, return_tet_mesh=True, ori_v=scaled_verts, num_tets_table=num_tets_table, tet_table=tet_table)
    
    # print(f"After tet marching with verts: {verts.size()}, faces: {faces.size()}")
    return verts, faces, sdf_values, GT_verts, GT_faces  # verts, faces #

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        # should save u here #
        # save_u_path = os.path.join("/data2/datasets/diffsim/neus/exp/hand_test/womask_sphere_reverse_value/other_saved", "sdf_values.npy")
        # np.save(save_u_path, u)
        # print(f"u saved to {save_u_path}")
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    
    ## using maching cubes ###
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold) # grid sdf and marching cubes #
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    ### using maching cubes ###
    
    ### using marching tets ###
    # vertices, triangles = extract_fields_from_tets(bound_min, bound_max, resolution, query_func)
    # vertices = vertices.detach().cpu().numpy()
    # triangles = triangles.detach().cpu().numpy()
    ### using marching tets ###
    
    # b_max_np = bound_max.detach().cpu().numpy()
    # b_min_np = bound_min.detach().cpu().numpy()

    # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def extract_geometry_tets(bound_min, bound_max, resolution, threshold, query_func, def_func=None):
    # print('threshold: {}'.format(threshold))
    
    ### using maching cubes ###
    # u = extract_fields(bound_min, bound_max, resolution, query_func)
    # vertices, triangles = mcubes.marching_cubes(u, threshold) # grid sdf and marching cubes #
    # b_max_np = bound_max.detach().cpu().numpy()
    # b_min_np = bound_min.detach().cpu().numpy()

    # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    ### using maching cubes ###
    
    ## 
    ### using marching tets ### fiels from tets ##
    vertices, triangles, tet_sdf_values,  GT_verts, GT_faces  = extract_fields_from_tets(bound_min, bound_max, resolution, query_func, def_func=def_func)
    # vertices = vertices.detach().cpu().numpy()
    # triangles = triangles.detach().cpu().numpy()
    ### using marching tets ###
    
    # b_max_np = bound_max.detach().cpu().numpy()
    # b_min_np = bound_min.detach().cpu().numpy()
    #

    # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles, tet_sdf_values,  GT_verts, GT_faces 


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF # invert cdf # 
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def load_GT_vertices(GT_meshes_folder):
    tot_meshes_fns = os.listdir(GT_meshes_folder)
    tot_meshes_fns = [fn for fn in tot_meshes_fns if fn.endswith(".obj")]
    tot_mesh_verts = []
    tot_mesh_faces = []
    n_tot_verts = 0
    for fn in tot_meshes_fns:
        cur_mesh_fn = os.path.join(GT_meshes_folder, fn)
        obj_mesh = trimesh.load(cur_mesh_fn, process=False)
        # obj_mesh.remove_degenerate_faces(height=1e-06)

        verts_obj = np.array(obj_mesh.vertices)
        faces_obj = np.array(obj_mesh.faces)
        
        tot_mesh_verts.append(verts_obj)
        tot_mesh_faces.append(faces_obj + n_tot_verts)
        n_tot_verts += verts_obj.shape[0]
        
        # tot_mesh_faces.append(faces_obj)
    tot_mesh_verts = np.concatenate(tot_mesh_verts, axis=0)
    tot_mesh_faces = np.concatenate(tot_mesh_faces, axis=0)
    return tot_mesh_verts, tot_mesh_faces


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        
        GT_meshes_folder = "/home/xueyi/diffsim/DiffHand/assets/hand"
        self.mesh_vertices, self.mesh_faces = load_GT_vertices(GT_meshes_folder=GT_meshes_folder)
        maxx_pts = 25.
        minn_pts = -15.
        self.mesh_vertices = (self.mesh_vertices - minn_pts) / (maxx_pts - minn_pts)
        f = SDF(self.mesh_vertices, self.mesh_faces)
        self.gt_sdf = f ## a unite sphere or box
        
        self.minn_pts = 0
        self.maxx_pts = 1.
        
        # self.minn_pts = -1.5 # gorudn-truth states with the deformation -> update the sdf value fiedl 
        # self.maxx_pts = 1.5
        self.bkg_pts = ... # TODO: the bkg pts # bkg_pts; # bkg_pts_defs #
        self.cur_fr_bkg_pts_defs = ... # TODO: set the cur_bkg_pts_defs for each frame #
        self.dist_interp_thres = ... # TODO: set the cur_bkg_pts_defs #
        
        self.bending_network = ... # TODO: add the bending network # 
        self.use_bending_network = ... # TODO: set the property #
        self.use_delta_bending = ... # TODO
        # use bending network # 
        
        
    # get the pts and render the pts #
    # pts and the rendering pts #
    def deform_pts(self, pts, pts_ts=0):
        
        if self.use_bending_network:
            if len(pts.size()) == 3:
                nnb, nns = pts.size(0), pts.size(1)
                pts_exp = pts.contiguous().view(nnb * nns, -1).contiguous()
            else:
                pts_exp = pts
            # pts_ts #
            if self.use_delta_bending:
                # if pts_ts >= 5:
                #     pts_exp = self.bending_network(pts_exp, input_pts_ts=pts_ts)
                #     for cur_pts_ts in range(4, -1, -1):
                #         # print(f"using delta bending with pts_ts: {cur_pts_ts}")
                #         pts_exp = self.bending_network(pts_exp, input_pts_ts=cur_pts_ts)
                # else:
                #     for cur_pts_ts in range(pts_ts, -1, -1):
                #         # print(f"using delta bending with pts_ts: {cur_pts_ts}")
                #         pts_exp = self.bending_network(pts_exp, input_pts_ts=cur_pts_ts)
                for cur_pts_ts in range(pts_ts, -1, -1):
                    # print(f"using delta bending with pts_ts: {cur_pts_ts}")
                    pts_exp = self.bending_network(pts_exp, input_pts_ts=cur_pts_ts)
            else:
                pts_exp = self.bending_network(pts_exp, input_pts_ts=pts_ts)
            if len(pts.size()) == 3:
                pts = pts_exp.contiguous().view(nnb, nns, -1).contiguous()
            else:
                pts = pts_exp
            return pts
            
        # pts: nn_batch x nn_samples x 3
        if len(pts.size()) == 3:
            nnb, nns = pts.size(0), pts.size(1)
            pts_exp = pts.contiguous().view(nnb * nns, -1).contiguous()
        else:
            pts_exp = pts
        # print(f"prior to deforming: {pts.size()}")
        
        dist_pts_to_bkg_pts = torch.sum(
            (pts_exp.unsqueeze(1) - self.bkg_pts.unsqueeze(0)) ** 2, dim=-1 ## nn_pts_exp x nn_bkg_pts
        )
        dist_mask = dist_pts_to_bkg_pts <= self.dist_interp_thres # 
        dist_mask_float = dist_mask.float()
        
        # dist_mask_float #
        cur_fr_bkg_def_exp = self.cur_fr_bkg_pts_defs.unsqueeze(0).repeat(pts_exp.size(0), 1, 1).contiguous()
        cur_fr_pts_def = torch.sum(
            cur_fr_bkg_def_exp * dist_mask_float.unsqueeze(-1), dim=1
        )
        dist_mask_float_summ = torch.sum(
            dist_mask_float, dim=1
        )
        dist_mask_float_summ = torch.clamp(dist_mask_float_summ, min=1)
        cur_fr_pts_def = cur_fr_pts_def / dist_mask_float_summ.unsqueeze(-1) # bkg pts deformation #
        pts_exp = pts_exp - cur_fr_pts_def
        if len(pts.size()) == 3:
            pts = pts_exp.contiguous().view(nnb, nns, -1).contiguous()
        else:
            pts = pts_exp
        return pts # 
    
    
        

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None, pts_ts=0):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints #
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3 #
        
        # pts = pts.flip((-1,)) * 2 - 1
        pts = pts * 2 - 1
        
        pts = self.deform_pts(pts=pts, pts_ts=pts_ts)

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4 #

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, pts_ts=0):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        
        # pts = pts.flip((-1,)) * 2 - 1
        pts = pts * 2 - 1
        
        pts = self.deform_pts(pts=pts, pts_ts=pts_ts)
        
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False, pts_ts=0):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        
        # pts = pts.flip((-1,)) * 2 - 1
        pts = pts * 2 - 1
        
        pts = self.deform_pts(pts=pts, pts_ts=pts_ts)
        
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    pts_ts=0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5 # z_vals and dists * 0.5 #

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3) # pts, nn_ou
        dirs = dirs.reshape(-1, 3)
        
        pts = (pts - self.minn_pts) / (self.maxx_pts - self.minn_pts)
        
        # pts = pts.flip((-1,)) * 2 - 1
        pts = pts * 2 - 1
        
        
        pts = self.deform_pts(pts=pts, pts_ts=pts_ts)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        # deviation network #
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, pts_ts=0, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, use_gt_sdf=False):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                
                pts = (pts - self.minn_pts) / (self.maxx_pts - self.minn_pts)
                # sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                # gt_sdf #
                
                # 
                # pts = ((pts - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
                
                # pts = pts.flip((-1,)) * 2 - 1
                pts = pts * 2 - 1
                
                pts = self.deform_pts(pts=pts, pts_ts=pts_ts)
                
                pts_exp = pts.reshape(-1, 3)
                # minn_pts, _ = torch.min(pts_exp, dim=0)
                # maxx_pts, _ = torch.max(pts_exp, dim=0) # deformation field (not a rigid one) -> the meshes #
                # print(f"minn_pts: {minn_pts}, maxx_pts: {maxx_pts}")

                # pts_to_near = pts - near.unsqueeze(1)
                # maxx_pts = 1.5; minn_pts = -1.5
                # # maxx_pts = 3; minn_pts = -3
                # # maxx_pts = 1; minn_pts = -1
                # pts_exp = (pts_exp - minn_pts) / (maxx_pts - minn_pts)
                
                ## render and iamges  ####  
                if use_gt_sdf:
                    ### use the GT sdf field ####
                    # print(f"Using gt sdf :")
                    sdf = self.gt_sdf(pts_exp.reshape(-1, 3).detach().cpu().numpy())
                    sdf = torch.from_numpy(sdf).float().cuda()
                    sdf = sdf.reshape(batch_size, self.n_samples)
                    ### use the GT sdf field ####
                else:
                    #### use the optimized sdf field ####
                    sdf = self.sdf_network.sdf(pts_exp).reshape(batch_size, self.n_samples)
                    #### use the optimized sdf field ####

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i,
                                                pts_ts=pts_ts)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  pts_ts=pts_ts)

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf, pts_ts=pts_ts)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o, # 
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    pts_ts=pts_ts)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min, # extract geometry #
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
        
    def extract_geometry_tets(self, bound_min, bound_max, resolution, pts_ts=0, threshold=0.0, wdef=False):
        if wdef:
            return extract_geometry_tets(bound_min, # extract geometry #
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts),
                                def_func=lambda pts: self.deform_pts(pts, pts_ts=pts_ts))
        else:
            return extract_geometry_tets(bound_min, # extract geometry #
                                    bound_max,
                                    resolution=resolution,
                                    threshold=threshold,
                                    query_func=lambda pts: -self.sdf_network.sdf(pts))
