import open3d as o3d
import numpy as np

def compute_fpfh_with_color(pcd, voxel_size):
    """
    Compute FPFH+Color features for a given point cloud.
    
    Parameters:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling and feature computation.
    
    Returns:
        pcd_down (open3d.geometry.PointCloud): Downsampled point cloud.
        fpfh_color_features (numpy.ndarray): Combined FPFH+Color features.
    """
    # Downsample the point cloud
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # Extract color information from the downsampled point cloud
    colors = np.asarray(pcd_down.colors)  # Shape: (N, 3), where N is the number of points

    # Combine FPFH features and color information
    fpfh_data = np.asarray(fpfh.data).T  # Transpose to shape (N, 33)
    fpfh_color_features = np.hstack((fpfh_data, colors))  # Combine FPFH and RGB

    return pcd_down, fpfh_color_features


# Path to the input point cloud file
pcd_path = '/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply'

# Load the point cloud
print(f"Loading point cloud from: {pcd_path}")
pcd = o3d.io.read_point_cloud(pcd_path)

# Check if the point cloud is loaded correctly
if not pcd.has_points():
    raise ValueError("Point cloud could not be loaded or is empty.")

# Set parameters and compute features
voxel_size = 0.05  # Adjust based on the dataset's scale
pcd_down, features = compute_fpfh_with_color(pcd, voxel_size)

# Display the extracted features
print("Extracted FPFH+Color features:")
print(features)

# Optionally save downsampled point cloud and features for further use
"""

o3d.io.write_point_cloud("downsampled_point_cloud.ply", pcd_down)
np.save("fpfh_color_features.npy", features)
"""