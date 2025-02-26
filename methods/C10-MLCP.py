import open3d as o3d
import numpy as np

def compute_mlcp_features(point_cloud, radii):
    """
    Compute Multiscale Local Color Pattern (MLCP) features for a 3D point cloud.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud.
        radii (list of float): List of radii for multiscale neighborhood computation.
    
    Returns:
        np.ndarray: Extracted MLCP features for each point in the cloud.
    """
    # Ensure the point cloud has colors
    if not point_cloud.has_colors():
        raise ValueError("Point cloud must have color information.")

    # Convert point cloud to numpy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    # KDTree for neighborhood search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    # Initialize feature storage
    mlcp_features = []

    # Iterate through each point in the point cloud
    for i, point in enumerate(points):
        local_features = []
        
        # Compute features at multiple scales
        for radius in radii:
            [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            neighbors_colors = colors[idx]
            
            # Compute local color pattern (mean and variance of colors)
            mean_color = np.mean(neighbors_colors, axis=0)
            var_color = np.var(neighbors_colors, axis=0)
            
            # Concatenate mean and variance as features
            local_features.extend(mean_color.tolist())
            local_features.extend(var_color.tolist())
        
        mlcp_features.append(local_features)

    return np.array(mlcp_features)

# Main script
if __name__ == "__main__":
    # Path to the input point cloud file
    point_cloud_path = '/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply'

    try:
        # Load the point cloud data
        pcd = o3d.io.read_point_cloud(point_cloud_path)

        # Verify if the point cloud was loaded successfully
        if not pcd.has_points():
            print("The point cloud is empty or could not be loaded.")
        else:
            print("Point cloud loaded successfully.")

            # Define radii for multiscale computation
            radii = [0.05, 0.1, 0.2]

            # Compute MLCP features
            mlcp_features = compute_mlcp_features(pcd, radii)

            print("Extracted MLCP Features:")
            print(mlcp_features)

    except Exception as e:
        print(f"An error occurred: {e}")
