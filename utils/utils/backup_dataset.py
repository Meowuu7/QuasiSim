import os
import shutil

def copy_folder_contents(source_folder, destination_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(source_folder):
        # 构建目标文件夹的对应子文件夹路径
        relative_path = os.path.relpath(root, source_folder)
        destination_dir = os.path.join(destination_folder, relative_path)

        # 检查目标文件夹是否存在，如果不存在则创建
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # 复制文件到目标文件夹中
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)

            # 检查目标文件是否存在，如果存在则跳过
            if not os.path.exists(destination_file):
                shutil.copy2(source_file, destination_file)

if __name__ == '__main__':
    date_list = ['20230915', '20230917', '20230927', '20230928']
    file_list = ['mano', 'src']
    for date in date_list:
        src_root = f'/share/hlyang/results/dataset/{date}'
        dst_root = f'/share/hlyang/results/dataset_old/{date}'
        os.makedirs(dst_root, exist_ok=True)
        video_list = os.listdir(src_root)
        video_list.sort()
        
        for video_id in video_list:
            try:
                src_video_root = os.path.join(src_root, video_id)
                for file in file_list:
                    assert os.path.exists(os.path.join(src_video_root, file))
                
                dst_video_root = os.path.join(dst_root, video_id)
                os.makedirs(dst_video_root, exist_ok=True)
                
                for file in file_list:
                    src_dir = os.path.join(src_video_root, file)
                    dst_dir = os.path.join(dst_video_root, file)
                    copy_folder_contents(src_dir, dst_dir)
            except:
                continue
                