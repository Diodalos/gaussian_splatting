import argparse
import open3d as o3d
import numpy as np
import os
from imu_convert import get_imu

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', '-s', required = True)
    parser.add_argument('--voxel_size', '-v', default=0.5, type=float)
    args = parser.parse_args()
    return args

def load_point_clouds(voxel_size):
    path = args.src_path
    files = os.listdir(path)
    pcds = []
    files.sort()

    for file in files:
        if "ply" in file:
            pcd = o3d.io.read_point_cloud(path + os.sep + file)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            pcds.append(pcd_down)

    return pcds

def radius_outlier_removal(pcds):
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

    return pcds

def pairwise_registration(source, target, odom_source, odom_target):
    radius_normal = args.voxel_size * 2
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # icp_coarse = o3d.pipelines.registration.registration_generalized_icp(
    #     source, target, max_correspondence_distance_coarse, np.identity(4),
    #     o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    # icp_fine = o3d.pipelines.registration.registration_generalized_icp(
    #     source, target, max_correspondence_distance_fine,
    #     icp_coarse.transformation,
    #     o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    icp_fine = o3d.pipelines.registration.registration_generalized_icp(
        source, target, max_correspondence_distance_fine,
        np.dot(np.linalg.inv(odom_source), odom_target),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, odoms):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], odoms[source_id], odoms[target_id])
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

if __name__ == '__main__':
    args = get_args()

    pcds_down = load_point_clouds(args.voxel_size)[100:110]
    pcds_down = radius_outlier_removal(pcds_down)

    o3d.visualization.draw_geometries(pcds_down,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    
    # max_correspondence_distance_coarse = args.voxel_size * 50
    # max_correspondence_distance_fine = args.voxel_size * 1.5

    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Error) as cm:
    #     pose_graph = full_registration(pcds_down,
    #                                 max_correspondence_distance_coarse,
    #                                 max_correspondence_distance_fine)
        
    # option = o3d.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0)
    
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Error) as cm:
    #     o3d.pipelines.registration.global_optimization(
    #         pose_graph,
    #         o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #         o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    #         option)

    poses = get_imu()[100:110]

    print(len(pcds_down))
    print(len(poses))

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        # print(pose_graph.nodes[point_id].pose)
        # pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(poses[point_id])
        pcd_combined += pcds_down[point_id]
    
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=args.voxel_size)
    o3d.io.write_point_cloud(args.src_path + os.sep + 'result' + os.sep + 'combined.ply', pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down],
                                    zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])