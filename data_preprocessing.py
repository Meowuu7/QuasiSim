import os
import numpy as np
import pickle
import torch

from manopth.manolayer import ManoLayer

def get_mano_model(ncomps=45, side='right'):
    # ncomps = 45 # mano root #
    batch_size = 1
    mano_root = "/home/xueyi/sim/manopth/mano/models"
    if not os.path.exists(mano_root):
      mano_root = "/root/diffsim/mano_v1_2/models"
    mano_model = ManoLayer(
        mano_root=mano_root, use_pca=True if ncomps == 15 else False, ncomps=ncomps, flat_hand_mean=False, side=side)
    return mano_model


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
        cur_faces = items[1:]
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


def extract_bimanual_hand_objs(seq_sv_fn, st_idx=42, window_size=60):
    obj_name = seq_sv_fn.split("/")[-1].split("_")[0]
    # obj_sv_path = "/Users/xymeow/Study/_2022_autumn/learning/taichi/exs/DiffHand/examples/save_res/object_vtemplates"
    obj_sv_path = "/home/xueyi/sim/arctic/data/arctic_data/data/meta/object_vtemplates"
    if not os.path.exists(obj_sv_path):
        obj_sv_path = "/data/xueyi/arctic/data/arctic_data/data/meta/object_vtemplates"
    obj_mesh_fn = os.path.join(obj_sv_path, obj_name, "mesh.obj")

    template_verts, template_faces = read_obj_file_ours(obj_mesh_fn, sub_one=True)
    mano_model = get_mano_model()
    hand_faces = mano_model.th_faces.numpy()

    seqs_data = np.load(seq_sv_fn, allow_pickle=True).item()
    rot_r = seqs_data['rot_r']
    trans_r = seqs_data['trans_r']
    pose_r = seqs_data['pose_r']
    shape_r = seqs_data['shape_r']
    mano_model_rgt = get_mano_model(ncomps=45, side='right')
    theta = np.concatenate([rot_r, pose_r], axis=-1)
    hand_verts, hand_joints = mano_model_rgt(torch.from_numpy(theta).float(), torch.from_numpy(shape_r).float(),
                                             torch.from_numpy(trans_r).float())  # hand
    hand_verts = hand_verts.numpy()
    hand_verts = hand_verts / 1000.

    hand_joints = hand_joints.numpy()

    rot_l = seqs_data['rot_l']
    trans_l = seqs_data['trans_l']
    pose_l = seqs_data['pose_l']
    shape_l = seqs_data['shape_l']
    mano_model_lft = get_mano_model(ncomps=45, side='left')
    theta = np.concatenate([rot_l, pose_l], axis=-1)
    hand_verts_lft, hand_joints_lft = mano_model_lft(torch.from_numpy(theta).float(), torch.from_numpy(shape_l).float(),
                                                     torch.from_numpy(trans_l).float()) # mano model left #
    hand_verts_lft = hand_verts_lft.numpy()
    hand_verts_lft = hand_verts_lft / 1000.

    obj_verts = seqs_data["verts.object"]
    # for fr in range(obj_verts.shape[0]):
    # for fr in range(100, obj_verts.shape[0]):

    tot_obj_verts = []
    tot_rhand_verts = []
    tot_lhand_verts = []

    for fr in range(st_idx, st_idx + window_size):
    # for fr in range(0, obj_verts.shape[0]):
        cur_obj_verts = obj_verts[fr]
        cur_rhand_verts = hand_verts[fr]
        cur_lhand_verts = hand_verts_lft[fr]
        # pcd = ps.register_point_cloud(f"rhand", cur_rhand_verts, radius=0.0052,
        #                               color=color[0 % len(color)])  ### color and color
        # pcd = ps.register_point_cloud(f"lhand", cur_lhand_verts, radius=0.0052,
        #                               color=color[2 % len(color)])  ### color and color
        # pcd = ps.register_point_cloud(f"obj", cur_obj_verts, radius=0.0052,
        #                               color=color[1 % len(color)])  ### color and color
        # # template_faces
        # obj_mesh = ps.register_surface_mesh(f"obj_mesh", cur_obj_verts, template_faces,
        #                                     color=color[1 % len(color)])
        # rhand_mesh = ps.register_surface_mesh(f"cur_rhand_mesh", cur_rhand_verts, hand_faces,
        #                                       color=color[0 % len(color)])
        # lhand_mesh = ps.register_surface_mesh(f"cur_lhand_mesh", cur_lhand_verts, hand_faces,
        #                                       color=color[2 % len(color)])
        # if fr == 0:
        #     ps.show()
        # ps.screenshot()
        # ps.remove_all_structures()

        tot_obj_verts.append(cur_obj_verts)
        tot_rhand_verts.append(cur_rhand_verts)
        tot_lhand_verts.append(cur_lhand_verts)
    tot_obj_verts = np.stack(tot_obj_verts, axis=0)
    tot_rhand_verts = np.stack(tot_rhand_verts, axis=0)
    tot_lhand_verts = np.stack(tot_lhand_verts, axis=0)

    obj_bbmin, obj_bbmax = tot_obj_verts.min(0).min(0), tot_obj_verts.max(0).max(0)
    tot_hand_verts = np.concatenate([tot_rhand_verts, tot_lhand_verts], axis=1) #
    hand_bbmin, hand_bbmax = tot_hand_verts.min(0).min(0), tot_hand_verts.max(0).max(0)
    tot_min = np.stack([obj_bbmin, hand_bbmin], axis=0).min(0)
    tot_max = np.stack([obj_bbmax, hand_bbmax], axis=0).max(0)
    center = (tot_min + tot_max) * 0.5 # center of the overall sequence #
    scale = 2.0 / (tot_max - tot_min).max()  # bounding box's max # # bbmax - bbmin #
    # vertices = (vertices - center) * scale  # (vertices - center) * scale #
    tot_obj_verts = (tot_obj_verts - np.reshape(center, (1, 1, 3))) * scale
    tot_rhand_verts = (tot_rhand_verts - np.reshape(center, (1, 1, 3))) * scale
    tot_lhand_verts = (tot_lhand_verts - np.reshape(center, (1, 1, 3))) * scale

    sv_dict = {
        'obj_verts': tot_obj_verts,
        'rhand_verts': tot_rhand_verts,
        'lhand_verts': tot_lhand_verts,
        'obj_faces': template_faces,
        'hand_faces': hand_faces,
        'st_idx': st_idx
    }
    sv_dict_root_folder = "/".join(seq_sv_fn.split("/")[:-1])
    seq_name = seq_sv_fn.split("/")[-1].split(".")[0]
    sv_dict_fn = os.path.join(sv_dict_root_folder, f"{seq_name}_extracted_dict.npy")
    np.save(sv_dict_fn, sv_dict)
    print(f"extracted values saved to {sv_dict_fn}")


# /data/datasets/genn/sim/arctic_processed_data/processed_seqs/s01/capsulemachine_grab_01_extracted_dict.npy
# /data/datasets/genn/sim/arctic_processed_data/processed_seqs/s01/ketchup_grab_01_extracted_dict.npy
if __name__=='__main__':
    
    seq_sv_fn = "/data/datasets/genn/sim/arctic_processed_data/processed_seqs"
    if not os.path.exists(seq_sv_fn):
        seq_sv_fn = "/data/xueyi/arctic/arctic_processed_data/processed_seqs"
    # seq_sv_fn = "/data/datasets/genn/sim/arctic_processed_data/processed_seqs"
    st_idx = 42
    window_size = 60
    
    tot_subjs = os.listdir(seq_sv_fn)
    for cur_subj in tot_subjs:
        print(f"processing subj: {cur_subj}")
        cur_subj_folder = os.path.join(seq_sv_fn, cur_subj)
        cur_subj_tot_seqs = os.listdir(cur_subj_folder)
        for cur_seq in cur_subj_tot_seqs:
            print(f"processing cur_seq: {cur_seq}")
            # ketchup_grab_01.npy
            cur_seq_obj_nm = cur_seq.split("_")[0]
            cur_seq_manip_type = cur_seq.split("_")[1]
            if cur_seq_manip_type == "grab":
                cur_process_seq_fn = os.path.join(cur_subj_folder, cur_seq)
                if cur_seq.endswith("extracted_dict.npy"):
                    continue
                extract_bimanual_hand_objs(cur_process_seq_fn, st_idx=st_idx, window_size=window_size)
            