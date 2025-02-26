import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def generate_intensity_color_spin_images(point_cloud, radius=0.1, bin_size=0.01):
    """
    Generate intensity- and color-based spin images for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud with color and intensity.
        radius (float): Support radius for spin image generation.
        bin_size (float): Bin size for the spin image.

    Returns:
        list: Spin images (2D histograms) for each point in the point cloud.
    """
    # Ensure the point cloud has normals
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()

    # Extract points, normals, colors, and intensities
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    colors = np.asarray(point_cloud.colors)

    # Compute intensity as the grayscale equivalent of RGB
    intensities = np.dot(colors, [0.2989, 0.5870, 0.1140])  # Standard grayscale conversion

    spin_images = []

    for i, (point, normal) in enumerate(zip(points, normals)):
        # Create a local coordinate system
        tangent = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.cross(normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)

        # Project neighbors into the local coordinate system
        distances = np.linalg.norm(points - point, axis=1)  # No reshaping needed
        neighbors = points[distances < radius]
        relative_positions = neighbors - point

        x_coords = np.dot(relative_positions, tangent)
        y_coords = np.dot(relative_positions, bitangent)

        # Combine intensity and color into a single feature vector
        neighbor_colors = colors[distances < radius]
        neighbor_intensities = intensities[distances < radius]
        combined_features = neighbor_intensities + np.mean(neighbor_colors, axis=1)

        # Create a weighted 2D histogram (spin image)
        hist, _, _ = np.histogram2d(
            x_coords,
            y_coords,
            bins=int(2 * radius / bin_size),
            range=[[-radius, radius], [-radius, radius]],
            weights=combined_features
        )
        
        spin_images.append(hist)

    return spin_images


# Load the specified PLY file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
pcd = o3d.io.read_point_cloud(file_path)

# Check if the point cloud was loaded successfully
if not pcd.has_points():
    raise ValueError(f"Failed to load point cloud from {file_path}")

# Generate spin images with intensity and color information
spin_images = generate_intensity_color_spin_images(pcd)

# Display one example spin image
plt.imshow(spin_images[0], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Example Intensity-Color Spin Image")
plt.show()

# Print summary of extracted features
print(f"Generated {len(spin_images)} spin images.")
