import open3d as o3d
import numpy as np

def compute_3d_shape_context_with_color(point_cloud, num_bins=5, radius=0.1):
    """
    Computes 3D Shape Context with Color (3DSC+Color) descriptors for a given point cloud.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud with colors.
        num_bins (int): Number of bins for the histogram in each dimension.
        radius (float): Radius for neighborhood search.
    
    Returns:
        descriptors (np.ndarray): Combined shape and color histograms for each point.
    """
    if not point_cloud.has_colors():
        raise ValueError("Point cloud must have color information.")

    # Convert to numpy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    # KDTree for neighborhood search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    
    descriptors = []
    
    for i, point in enumerate(points):
        # Find neighbors within the radius
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        
        if len(idx) < 2:  # Skip isolated points
            descriptors.append(np.zeros((num_bins**3 + num_bins**3)))
            continue
        
        # Extract neighbors' positions and colors
        neighbors = points[idx]
        neighbor_colors = colors[idx]
        
        # Compute relative positions
        relative_positions = neighbors - point
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Compute spherical coordinates (r, theta, phi)
        r = distances
        theta = np.arccos(relative_positions[:, 2] / (r + 1e-8))
        phi = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        
        # Normalize and bin spatial features
        r_bin_edges = np.linspace(0, radius, num_bins + 1)
        theta_bin_edges = np.linspace(0, np.pi, num_bins + 1)
        phi_bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        
        spatial_histogram, _ = np.histogramdd(
            (r, theta, phi),
            bins=(r_bin_edges, theta_bin_edges, phi_bin_edges)
        )
        
        # Flatten spatial histogram
        spatial_histogram = spatial_histogram.flatten()
        
        # Normalize and bin color features
        color_bin_edges = np.linspace(0, 1, num_bins + 1)
        color_histogram_r, _ = np.histogram(neighbor_colors[:, 0], bins=color_bin_edges)
        color_histogram_g, _ = np.histogram(neighbor_colors[:, 1], bins=color_bin_edges)
        color_histogram_b, _ = np.histogram(neighbor_colors[:, 2], bins=color_bin_edges)
        
        # Concatenate histograms into a single descriptor
        color_histogram = np.concatenate([color_histogram_r, color_histogram_g, color_histogram_b])
        
        descriptor = np.concatenate([spatial_histogram, color_histogram])
        
        descriptors.append(descriptor)
    
    return np.array(descriptors)

# Load the point cloud from the specified path
point_cloud_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
pcd = o3d.io.read_point_cloud(point_cloud_path)

# Check if the point cloud is loaded successfully
if not pcd.has_points():
    raise ValueError("Failed to load the point cloud or it contains no points.")

# Compute the descriptors
descriptors = compute_3d_shape_context_with_color(pcd)

# Print the shape of the extracted features
print("Extracted Features Shape:", descriptors.shape)

# Visualize the point cloud with Open3D
#o3d.visualization.draw_geometries([pcd])
