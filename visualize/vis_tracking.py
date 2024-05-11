import random

import argparse

try:
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
    ps.set_screenshot_extension(".png")
except:
    pass

print("here1")
from scipy.spatial.transform import Rotation as R

import numpy as np


import sys
sys.path.append("./manopth")
from manopth.manolayer import ManoLayer
import torch
color = [
    (136/255.0,224/255.0,239/255.0),
    (180/255.0,254/255.0,152/255.0),
    (184/255.0,59/255.0,94/255.0),
    (106/255.0,44/255.0,112/255.0),
    (39/255.0,53/255.0,135/255.0),
(0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0), (234/255.0,255/255.0,208/255.0),(162/255.0,210/255.0,255/255.0),
    (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (184/255.0,59/255.0,94/255.0),(106/255.0,44/255.0,112/255.0),(39/255.0,53/255.0,135/255.0),
]

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

gray_color = (233 / 255., 241 / 255., 148 / 255.)


def vis_mano_tracking(tracking_info_fn):
    tracking_info = np.load(tracking_info_fn, allow_pickle=True).item()
    ts_to_retargeted_info = tracking_info['ts_to_retargeted_info']
    ## from the retargeted info and the retargeted info ##
    ## 
    ## get the hand faces and the obj_faces ##
    if 'obj_faces' in tracking_info:
        obj_faces = tracking_info['obj_faces']
    elif 'obj_faces_np'  in tracking_info:
        obj_faces = tracking_info['obj_faces_np']
    else:
        obj_faces = None
    
    if 'hand_faces' in tracking_info:
        hand_faces = tracking_info['hand_faces']
    else:
        hand_faces = None
    
    ## hand faces and the obj faces ##
    for ts in ts_to_retargeted_info:
        cur_ts_info = ts_to_retargeted_info[ts]
        ## current ts information here ##
        cur_hand_verts, cur_mano_rhand,  obj_verts = cur_ts_info
        ## hand faces and the obj faces ##
        if hand_faces is not None:
            ps.register_surface_mesh(f"hand", cur_hand_verts, hand_faces,
                                             color=color[0 % len(color)])  # gray_color
        else:
            ps.register_point_cloud(f"hand", cur_hand_verts, radius=0.012, color=color[0 % len(color)])
        if obj_faces is not None:
            ps.register_surface_mesh(f"object", obj_verts, obj_faces,
                                             color=gray_color)  # gray_color
        else:
            ps.register_point_cloud(f"object", obj_verts, radius=0.012, color=gray_color)
        ps.show()
        ps.set_screenshot_extension(".jpg")
        ps.screenshot()
        ps.remove_all_structures()


if __name__=='__main__':
    
    # python visualize/vis_tracking.py --tracking_info_fn=${saved_tracking_info_fn}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_info_fn', type=str, default='./save_res/retargeting_info_00001260.npy')
    args = parser.parse_args()
    
    # 'retargeting_info_{:0>8d}.npy' #
    
    tracking_info_fn = args.tracking_info_fn 
    vis_mano_tracking(tracking_info_fn=tracking_info_fn)
    
        