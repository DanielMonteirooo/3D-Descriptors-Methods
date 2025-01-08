import open3d as o3d
import numpy as np

def compute_color_shot_features(point_cloud, radius=0.05):
    """
    Compute Color-SHOT features for a given point cloud.

    Args:
        point_cloud (open3d.geometry.PointCloud): Input point cloud with colors.
        radius (float): Radius to consider for SHOT feature computation.

    Returns:
        numpy.ndarray: Extracted Color-SHOT features.
    """
    if not point_cloud.has_colors():
        raise ValueError("The input point cloud must have color information.")

    # Estimate normals if not already computed
    if not point_cloud.has_normals():
        print("Estimating normals for the point cloud...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )

    # Create the SHOTColorEstimation object
    shot_estimation = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )

    # Extract the features
    features = np.array(shot_estimation.data).T

    return features

# Example Usage
if __name__ == "__main__":
    # Load a sample point cloud with colors
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)

    # Ensure the point cloud has color information
    if not pcd.has_colors():
        print("Adding dummy color information for demonstration purposes.")
        colors = np.random.rand(np.asarray(pcd.points).shape[0], 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Compute Color-SHOT features
    print("Computing Color-SHOT features...")
    color_shot_features = compute_color_shot_features(pcd, radius=0.05)

    # Show the extracted features
    print("Extracted Color-SHOT features:")
    print(color_shot_features)
    print(f"Number of features: {color_shot_features.shape[0]}")
    print(f"Feature dimension: {color_shot_features.shape[1]}")
                                                    