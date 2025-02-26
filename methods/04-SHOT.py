import open3d as o3d
import numpy as np

def compute_shot_features(point_cloud, radius):
    """
    Compute SHOT (Signature of Histograms of Orientations) descriptors for a point cloud.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud.
        radius (float): Radius for the SHOT descriptor computation.

    Returns:
        numpy.ndarray: SHOT descriptors for each keypoint in the point cloud.
    """
    # Ensure the point cloud has normals computed
    if not point_cloud.has_normals():
        print("Computing normals for the point cloud...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Create a KDTree for neighborhood search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    # Placeholder for SHOT descriptors
    shot_descriptors = []

    # Iterate over all points in the cloud
    for i in range(len(point_cloud.points)):
        # Find neighbors within the radius
        _, idx, _ = kdtree.search_radius_vector_3d(point_cloud.points[i], radius)
        
        if len(idx) < 5:  # Skip if not enough neighbors
            shot_descriptors.append(np.zeros(352))  # Placeholder for empty descriptor
            continue

        # Compute local reference frame (LRF)
        neighbors = np.asarray(point_cloud.points)[idx, :]
        normals = np.asarray(point_cloud.normals)[idx, :]
        
        # Use PCA to determine LRF axes
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        # Compute histograms based on angles between normals and LRF axes
        reference_normal = np.asarray(point_cloud.normals)[i]
        cos_theta = np.dot(normals, reference_normal)
        
        # Bin the cos(theta) values into a histogram
        hist, _ = np.histogram(cos_theta, bins=11, range=(-1.0, 1.0))
        
        # Normalize histogram to create the descriptor
        descriptor = hist / np.linalg.norm(hist)
        shot_descriptors.append(descriptor)

    return np.array(shot_descriptors)


# Path to your PLY file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the point cloud from the provided path
print(f"Loading point cloud from: {file_path}")
pcd = o3d.io.read_point_cloud(file_path)

if not pcd.is_empty():
    print("Point cloud successfully loaded.")

    # Compute SHOT features
    print("Computing SHOT features...")
    radius = 0.05  # Set an appropriate radius based on your data scale
    shot_features = compute_shot_features(pcd, radius)

    # Display results
    print("SHOT Features (first 5 descriptors):")
    print(shot_features[:5])  # Display only the first 5 descriptors for brevity

    # Visualize the point cloud with Open3D
    """
   
    o3d.visualization.draw_geometries([pcd])
else:
    print("Failed to load the point cloud. Please check the file path.")
    """
