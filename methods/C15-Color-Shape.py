import open3d as o3d
import numpy as np

def graph_based_color_shape_descriptors(point_cloud, radius=0.1, max_nn=30):
    """
    Computes graph-based color-shape descriptors for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        radius (float): Radius for nearest neighbor search when estimating normals.
        max_nn (int): Maximum number of neighbors to consider.

    Returns:
        np.ndarray: Combined feature matrix of shape (N, 4), where each row contains
                    [R, G, B, Curvature].
    """
    # Step 1: Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    # Step 2: Compute curvature as a shape descriptor
    curvatures = np.asarray(point_cloud.compute_nearest_neighbor_distance())
    curvatures = curvatures / np.max(curvatures)  # Normalize curvature values
    
    # Step 3: Retrieve color information
    colors = np.asarray(point_cloud.colors)
    
    if colors.shape[0] == 0:
        raise ValueError("Point cloud does not contain color information.")
    
    # Step 4: Combine color and curvature into a feature matrix
    features = np.hstack((colors, curvatures.reshape(-1, 1)))
    
    return features

# Path to the point cloud file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the point cloud from the specified path
point_cloud = o3d.io.read_point_cloud(file_path)

# Check if the point cloud is loaded successfully
if not point_cloud.has_points():
    raise ValueError("Failed to load the point cloud or the file is empty.")

# Extract features using the implemented function
features = graph_based_color_shape_descriptors(point_cloud)

# Display the first few feature vectors
print("Extracted Features (first 5 rows):")
print(features[:5])
