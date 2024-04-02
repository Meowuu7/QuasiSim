import os
import sys
sys.path.append('.')
import numpy as np
from utils.process_frame_loss2 import cal_common_timestamps

def load_Luster_timestamps(file_path):
    ts = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            ts.append(int(line.split(" ")[-1]))
    return ts

def parse_trc(trc_paths):
    data_list = []
    for trc_path in trc_paths:
        cnt = 0
        data = {
            "timestamps": [],
            "markers": [],
        }
        N_marker = None
        with open(trc_path, "r") as f:
            for line in f:
                cnt += 1
                line = line.strip()
                if cnt == 4:
                    while line.find("\t\t") > -1:
                        line = line.replace("\t\t", "\t")
                    N_marker = len(line.split("\t")) - 3
                    if N_marker == 0:
                        N_marker = 10
                    # print("[parse_trc] file {}: N_marker = {}".format(trc_path, N_marker))
                if cnt <= 6:
                    continue
                # assert N_marker >= 3
                values = line.split("\t")
                data["timestamps"].append(int(values[2]))
                markers = np.ones((N_marker, 3)).astype(np.float32) * 10000
                for i in range(3, len(values), 3):
                    x, y, z = values[i : i + 3]
                    if len(x) > 0:
                        markers[(i//3) - 1, 0] = float(x) / 1000
                        markers[(i//3) - 1, 1] = float(y) / 1000
                        markers[(i//3) - 1, 2] = float(z) / 1000
                data["markers"].append(markers)
    
        data["timestamps"] = np.uint64(data["timestamps"])
        data["markers"] = np.float32(data["markers"])
        
        data_list.append(data)
            
    return data_list

def trc_to_timestamps(trc_data):
    NOKOV_ts = trc_data[0]["timestamps"]
    day_after_20230829 = (NOKOV_ts.min() - 1693238400000) // 86400000
    NOKOV_ts = list((NOKOV_ts - 1693238400000 - day_after_20230829 * 86400000).astype(np.int64))
    return NOKOV_ts

if __name__ == '__main__':
    
    # upload_root = '/data2/HOI-mocap'
    upload_root = '/share/datasets/HOI-mocap'
    date = '20230930'
    
    # for i in range(1, 153):
    #     try:
    #         trc_video_id = f'{date}_{str(i).zfill(3)}'
    #         rgb_video_id = f'{date}_{str(i).zfill(3)}'
        
    trc_video_id = '20230930_002'
    
    nokov_dir = os.path.join(upload_root, date, trc_video_id, 'nokov')
    trc_path = os.path.join(nokov_dir, [file for file in os.listdir(nokov_dir) if file.endswith('.trc')][0])
    # trc_path = '/data2/HOI-mocap/20231027/20231027_002/nokov/20231027_178_039_1-obj_039.trc'
    
    trc_data = parse_trc([trc_path])
    # print(trc_data[0]['timestamps'])
    nokov_ts = trc_to_timestamps(trc_data)
    
    rgb_video_id = '20230930_001'
    rgb_ts_path = os.path.join(upload_root, date, rgb_video_id, 'src', 'common_timestamp.txt')
    rgb_ts = load_Luster_timestamps(rgb_ts_path)
    # print(rgb_ts)
    
    ts_list = [nokov_ts, rgb_ts]
    
    error_threshold = 16
    common_timestamps = cal_common_timestamps(ts_list, error_threshold)
    
    print(trc_video_id, len(common_timestamps))
            # print(common_timestamps)
        # except:
        #     continue
    