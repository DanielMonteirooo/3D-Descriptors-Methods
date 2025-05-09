import numpy as np
import open3d as o3d

def compute_rgb_covariance_descriptor(point_cloud):
    """
    Compute the RGB Covariance Descriptor for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud with color information.

    Returns:
        np.ndarray: Covariance matrix representing the RGB Covariance Descriptor.
    """
    # Extract points and colors
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Combine points and colors into a single feature matrix
    features = np.hstack((points, colors))

    # Compute the mean of the features
    mean_features = np.mean(features, axis=0)

    # Center the features by subtracting the mean
    centered_features = features - mean_features

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_features, rowvar=False)

    return np.linalg.eigvals(np.nan_to_num(covariance_matrix))
# Path to your point cloud file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the point cloud from the specified file path
pcd = o3d.io.read_point_cloud(file_path)

# Ensure the point cloud has color information
if not pcd.has_colors():
    raise ValueError("The provided point cloud does not have color information.")

# Compute the RGB Covariance Descriptor
rgb_covariance_descriptor = compute_rgb_covariance_descriptor(pcd)

# Display the descriptor
print("RGB Covariance Descriptor:")
print(rgb_covariance_descriptor)
