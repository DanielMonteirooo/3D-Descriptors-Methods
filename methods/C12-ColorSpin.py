import numpy as np
import open3d as o3d

def compute_color_spin_descriptor(point_cloud, voxel_size=0.05, support_radius=0.1):
    """
    Computes a color-enhanced spin descriptor for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling.
        support_radius (float): Radius for neighborhood computation.

    Returns:
        descriptors (np.ndarray): Color-enhanced spin descriptors for each point.
    """
    # Downsample the point cloud
    pcd_down = point_cloud.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=2 * voxel_size, max_nn=30))

    # Convert RGB to intensity
    colors = np.asarray(pcd_down.colors)
    intensities = np.linalg.norm(colors, axis=1)

    # Compute spin image-like descriptors
    points = np.asarray(pcd_down.points)
    normals = np.asarray(pcd_down.normals)
    descriptors = []

    for i, (point, normal) in enumerate(zip(points, normals)):
        # Define local coordinate system
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Transform neighbors into local frame
        distances = np.linalg.norm(points - point[None, :], axis=1)  # Ensure proper broadcasting
        neighbor_indices = distances < support_radius               # Find neighbors within support radius
        neighbors = points[neighbor_indices]
        
        relative_positions = neighbors - point
        local_coords = np.dot(relative_positions, np.vstack([x_axis, y_axis, z_axis]).T)

        # Create 2D histogram (spin image) with intensity as an extra dimension
        alpha = np.sqrt(local_coords[:, 0]**2 + local_coords[:, 1]**2)
        beta = local_coords[:, 2]
        hist_2d, _, _ = np.histogram2d(alpha, beta, bins=(10, 10), range=[[0, support_radius], [-support_radius, support_radius]])

        # Add color intensity information
        local_intensities = intensities[neighbor_indices]
        hist_intensity = np.histogram(local_intensities, bins=10, range=(0, 1))[0]

        # Combine shape and color histograms
        descriptor = np.hstack([hist_2d.flatten(), hist_intensity])
        descriptors.append(descriptor)

    return np.array(descriptors)

# Load the point cloud from the specified path
pcd_path = '/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply'
pcd = o3d.io.read_point_cloud(pcd_path)

# Check if the point cloud is loaded correctly
if not pcd.has_points():
    raise ValueError("Failed to load point cloud. Please check the file path.")

# Compute descriptors
descriptors = compute_color_spin_descriptor(pcd)

# Visualize the extracted features
print("Extracted Color Spin Descriptors:")
print(descriptors)

#Error