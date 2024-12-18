import open3d as o3d
import numpy as np

def compute_3d_hough_transform(point_cloud, voxel_size=0.05):
    """
    Computes the 3D Hough Transform for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Size of the voxels for discretization.

    Returns:
        np.ndarray: A 3D array representing the Hough space (accumulator).
    """
    # Get the point cloud bounds
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()

    # Compute voxel grid dimensions
    dimensions = ((max_bound - min_bound) / voxel_size).astype(int)

    # Initialize Hough accumulator
    hough_space = np.zeros(dimensions, dtype=int)

    # Get the point cloud as a numpy array
    points = np.asarray(point_cloud.points)

    # Transform each point into voxel space
    for point in points:
        voxel_idx = ((point - min_bound) / voxel_size).astype(int)
        hough_space[tuple(voxel_idx)] += 1

    return hough_space

def extract_hough_features(hough_space, threshold=1):
    """
    Extracts features from the 3D Hough space based on a threshold.

    Parameters:
        hough_space (np.ndarray): The 3D Hough space (accumulator).
        threshold (int): Minimum number of votes to consider a feature.

    Returns:
        list: List of voxel indices corresponding to detected features.
    """
    features = np.argwhere(hough_space >= threshold)
    return features

# Load example point cloud
pcd = o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)

# Compute 3D Hough Transform
voxel_size = 0.05
hough_space = compute_3d_hough_transform(pcd, voxel_size)

# Extract features from Hough space
threshold = 5
features = extract_hough_features(hough_space, threshold)

# Print extracted features
print("Extracted features (voxel indices):")
print(features)

# Visualize the point cloud and features
voxel_indices = [tuple(f) for f in features]
voxel_points = [np.array(voxel_idx) * voxel_size + pcd.get_min_bound() for voxel_idx in voxel_indices]

# Create feature point cloud
features_pcd = o3d.geometry.PointCloud()
features_pcd.points = o3d.utility.Vector3dVector(voxel_points)
features_pcd.paint_uniform_color([1, 0, 0])

# Visualize both original point cloud and features
pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([pcd, features_pcd],
                                  window_name="3D Hough Transform Features",
                                  width=800,
                                  height=600,
                                  left=50,
                                  top=50)
