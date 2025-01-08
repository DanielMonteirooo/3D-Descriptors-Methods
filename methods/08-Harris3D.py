import numpy as np
import open3d as o3d

def harris_3d_detection(point_cloud, k=0.04, threshold=1e-6):
    # Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # Compute covariance matrices for each point's neighborhood
    covariances = []
    for i in range(len(point_cloud.points)):
        [k, idx, _] = point_cloud.tree.search_knn_vector_3d(point_cloud.points[i], 30)
        if k < 3:
            covariances.append(np.zeros((3, 3)))
            continue
        neighbors = np.asarray(point_cloud.points)[idx, :]
        cov = np.cov(neighbors.T)
        covariances.append(cov)

    # Compute Harris response
    harris_response = []
    for cov in covariances:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)
        response = eigvals[0] * eigvals[1] - k * (eigvals[0] + eigvals[1]) ** 2
        harris_response.append(response)

    # Thresholding to find interest points
    harris_response = np.array(harris_response)
    interest_points = np.where(harris_response > threshold)[0]

    return interest_points

# Load your point cloud
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/models/ricardo9/ply/frame0000.ply")

# Detect Harris 3D features
interest_points_idx = harris_3d_detection(pcd)

# Extract interest points
interest_points = pcd.select_by_index(interest_points_idx)

# Visualize
pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Original point cloud in gray
interest_points.paint_uniform_color([1, 0, 0])  # Interest points in red
o3d.visualization.draw_geometries([pcd, interest_points])
