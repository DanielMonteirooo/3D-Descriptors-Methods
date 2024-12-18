import open3d as o3d
import numpy as np

def compute_rsd(point_cloud, radius):
    """
    Compute Radius-based Surface Descriptors (RSD) for a point cloud.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud.
        radius (float): Radius for neighborhood search.
        
    Returns:
        np.ndarray: RSD feature vectors for each point.
    """
    # Estimate normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=30))

    # Compute distances between neighbors
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    rsd_features = []

    for i, point in enumerate(point_cloud.points):
        # Perform radius search to find neighbors
        _, idx, _ = kd_tree.search_radius_vector_3d(point, radius)
        if len(idx) < 3:
            # Skip if not enough neighbors
            rsd_features.append([0, 0])
            continue
        
        # Compute distances to neighbors
        neighbors = np.asarray(point_cloud.points)[idx, :]
        distances = np.linalg.norm(neighbors - point, axis=1)

        # Compute mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Append RSD features (mean and std of distances)
        rsd_features.append([mean_distance, std_distance])

    return np.array(rsd_features)

# Example Usage
if __name__ == "__main__":
    # Load a point cloud file
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/models/frame0000.pcd")

    # Set radius for RSD computation
    search_radius = 0.1  # Adjust based on the scale of your data

    # Compute RSD features
    rsd_features = compute_rsd(pcd, search_radius)

    # Display the first few RSD features
    print("First 5 RSD features:")
    print(rsd_features[:5])

    # Optional: Visualize the point cloud with normals
    o3d.visualization.draw_geometries([pcd])
