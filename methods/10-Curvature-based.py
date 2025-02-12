import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  #linha para importar o matplotlib.pyplot

def compute_curvature(point_cloud, radius=0.1, max_nn=30):
    """
    Computes curvature for each point in the point cloud.

    Parameters:
    - point_cloud (open3d.geometry.PointCloud): The input point cloud.
    - radius (float): Radius for neighborhood search.
    - max_nn (int): Maximum number of neighbors to consider.

    Returns:
    - curvatures (numpy.ndarray): Curvature values for each point.
    """
    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # Estimate covariance matrices
    covariances = o3d.geometry.PointCloud.estimate_point_covariances(
        point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    curvatures = np.zeros(len(covariances))

    for i, cov in enumerate(covariances):
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        # Sort eigenvalues to ensure correct order
        eigenvalues = np.sort(eigenvalues)
        # Compute curvature (ratio of smallest eigenvalue to sum of eigenvalues)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)

    return curvatures

# Load point cloud
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")

# Compute curvature
curvatures = compute_curvature(pcd)

# Print first 10 curvature values
print("Curvature values for the first 10 points:")
print(curvatures[:10])

# Visualize curvature
# Normalize curvature values for visualization
curvatures_normalized = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))
colors = plt.get_cmap('viridis')(curvatures_normalized)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize point cloud with curvature-based coloring
o3d.visualization.draw_geometries([pcd])
