import argparse
import open3d as o3d
import numpy as np
import struct
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', '-s', required = True)
    parser.add_argument('--dest_path', '-m', required = True)
    parser.add_argument('--size_float', default = 4, type=int)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    args = get_args()  
        
    if os.path.exists(args.dest_path) == False:
        os.makedirs(args.dest_path)
    
    files_to_open = os.listdir(args.src_path)
    for file_to_open in files_to_open:
        
        print("write : ", file_to_open)

        np_pcd = np.fromfile(args.src_path + os.sep + file_to_open, dtype=np.float32).reshape((-1, 4))
        np_pcd = np_pcd[:, :3]

        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(np_pcd)

        file_to_save = args.dest_path + os.sep + os.path.splitext(file_to_open)[0] + ".ply"
        
        o3d.io.write_point_cloud(file_to_save, pcd)