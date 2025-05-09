import numpy as np
import open3d as o3d

def compute_covariance_matrices(pcd, knn=30):
    """Compute covariance matrices using proper KDTree initialization"""
    # Create KDTreeFlann instance from point cloud geometry
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    covariances = []
    for i, point in enumerate(pcd.points):
        # Use the tree instance for search operations
        [k, idx, _] = tree.search_knn_vector_3d(point, knn)
        if k < 3:
            covariances.append(np.zeros((3, 3)))
            continue
        neighbors = np.asarray(pcd.points)[idx]
        cov = np.cov(neighbors.T)
        covariances.append(cov)
    return covariances

def harris_3d_response(covariances, k=0.04):
    """Calculate Harris 3D response from covariance matrices"""
    responses = []
    for cov in covariances:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals.sort()
        response = eigvals[0] * eigvals[1] - k * (eigvals[0] + eigvals[1])**2
        responses.append(response)
    return np.array(responses)

def detect_harris_3d_features(pcd, k=0.04, threshold=1e-6, knn=30):
    """Main detection function implementing Harris 3D algorithm"""
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=knn))
    covariances = compute_covariance_matrices(pcd, knn)
    responses = harris_3d_response(covariances, k)
    interest_idx = np.where(responses > threshold)[0]
    return interest_idx, responses

def calculate_statistical_averages(points):
    """Calculate required statistical averages from point coordinates"""
    row_means = np.mean(points, axis=1)
    col_means = np.mean(points, axis=0)
    avg_row = np.mean(row_means)
    avg_col = np.mean(col_means)
    return (avg_row + avg_col) / 2

# Pipeline execution
if __name__ == "__main__":
    # Load and process point cloud
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")
    interest_idx, responses = detect_harris_3d_features(pcd)
    interest_points = pcd.select_by_index(interest_idx)
    
    # Visualization setup
    base_cloud = pcd.paint_uniform_color([0.5, 0.5, 0.5])
    interest_points.paint_uniform_color([1, 0, 0])
    #o3d.visualization.draw_geometries([base_cloud, interest_points])
    
    # Statistical calculations
    interest_coords = np.asarray(interest_points.points)
    combined_average = calculate_statistical_averages(interest_coords)
    print(f"Combined statistical average: {combined_average:.4f}")
