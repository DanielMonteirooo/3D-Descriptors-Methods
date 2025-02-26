# Step 1: Install Open3D if not installed
# Run this command in your environment if Open3D is not installed: !pip install open3d

import open3d as o3d
import numpy as np

def compute_horn_features(point_cloud):
    """
    Compute the Histogram of RGB Normals (HoRN) features for a given point cloud.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud with RGB colors.

    Returns:
        np.ndarray: Feature vector representing the HoRN descriptor.
    """
    # Estimate normals for the point cloud
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Normalize RGB values to [0, 1]
    colors = np.asarray(point_cloud.colors)
    if colors.size == 0:
        raise ValueError("Point cloud does not contain color information.")
    
    normals = np.asarray(point_cloud.normals)

    # Compute RGB normals by multiplying RGB values with normals
    rgb_normals = colors * normals

    # Create histograms for each channel (R, G, B)
    histograms = []
    for i in range(3):  # Iterate over R, G, B channels
        hist, _ = np.histogram(rgb_normals[:, i], bins=10, range=(-1, 1))
        histograms.append(hist)

    # Concatenate histograms into a single feature vector
    feature_vector = np.concatenate(histograms)
    
    return feature_vector

# Load the point cloud from the specified path
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
try:
    pcd = o3d.io.read_point_cloud(file_path)
except Exception as e:
    raise FileNotFoundError(f"Could not load point cloud from {file_path}. Error: {e}")

# Check if the point cloud has color information
if not pcd.has_colors():
    raise ValueError("The input point cloud does not have RGB color information.")

# Compute HoRN features
horn_features = compute_horn_features(pcd)

# Display the extracted features
print("Extracted HoRN Features:")
print(horn_features)
