import open3d as o3d
import numpy as np

def compute_usc_features(point_cloud, voxel_size=0.05, search_radius=0.1):
    # Downsample the point cloud
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)

    # Estimate normals
    downsampled_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, max_nn=30))

    # Compute FPFH features as an approximation to 3DSC
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, max_nn=100))

    # Normalize FPFH features to approximate rotational invariance
    fpfh.data = fpfh.data / np.linalg.norm(fpfh.data, axis=0)

    return fpfh

# Load a sample point cloud
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/models/ricardo9/ply/frame0000.ply")

# Compute USC features
usc_features = compute_usc_features(pcd)

# Display the computed features
print(usc_features)
