import open3d as o3d
import numpy as np

def clustered_color_distribution_descriptor(point_cloud, voxel_size=0.05, bins=8):
    """
    Implements the Clustered Color Distribution Descriptor (CCDD).
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Size of the voxel for downsampling.
        bins (int): Number of bins for the RGB histogram.
    
    Returns:
        np.ndarray: Normalized color histogram as a feature descriptor.
    """
    # Step 1: Downsample the point cloud
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    # Step 2: Extract colors from the downsampled point cloud
    colors = np.asarray(downsampled_pcd.colors)
    
    if colors.size == 0:
        raise ValueError("The point cloud does not contain color information.")

    # Step 3: Compute a 3D histogram in RGB space
    hist, edges = np.histogramdd(colors, bins=(bins, bins, bins), range=((0, 1), (0, 1), (0, 1)))

    # Step 4: Normalize the histogram to create the descriptor
    hist_normalized = hist / np.sum(hist)

    return hist_normalized

# Path to your point cloud file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the point cloud from the specified path
point_cloud = o3d.io.read_point_cloud(file_path)

# Check if the point cloud is loaded successfully
if not point_cloud.has_colors():
    raise ValueError("The loaded point cloud does not have color information.")

# Extract features using the CCDD method
features = clustered_color_distribution_descriptor(point_cloud)

# Display the extracted features
print("Extracted CCDD Features:")
print(features)
