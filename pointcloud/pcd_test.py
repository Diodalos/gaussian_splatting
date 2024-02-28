import argparse
import open3d as o3d
import numpy as np
import os

def load_point_clouds(voxel_size):
    path = './data/kitti/2011_09_26_drive_0005_sync/ply_points'
    files = os.listdir(path)
    pcds = []
    files.sort()

    for file in files:
        if "ply" in file:
            pcd = o3d.io.read_point_cloud(path + os.sep + file)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            pcds.append(pcd_down)

    return pcds

pcds = load_point_clouds(0.5)[100:110]

for id, pcd in enumerate(pcds):
        # cloud, inliers = pcd.remove_radius_outlier(nb_points=5, radius=args.voxel_size * 2)
        # inlier_cloud = pcd.select_by_index(inliers)
        # outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        # outlier_cloud.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        # pcd = pcd.select_by_index(inliers)

    plane_model, road_inliers = pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=100)
        # road_cloud = pcd.select_by_index(road_inliers)
        # other_cloud = pcd.select_by_index(road_inliers, invert=True)
        # road_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        # other_cloud.paint_uniform_color([1,0,0])
        # o3d.visualization.draw_geometries([road_cloud, other_cloud])
    pcd = pcd.select_by_index(road_inliers, invert=True)
    pcds[id] = pcd

o3d.visualization.draw_geometries(pcds,
                                zoom=0.3412,
                                front=[0.4257, -0.2125, -0.8795],
                                lookat=[2.6172, 2.0475, 1.532],
                                up=[-0.0694, -0.9768, 0.2024])

