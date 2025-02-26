import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_rgbd_histograms_from_point_cloud(pcd, num_bins=32):
    """
    Compute RGB-D histograms for a given point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud with RGB and depth information.
        num_bins (int): Number of bins for histograms.
    
    Returns:
        np.ndarray: Concatenated histogram features for RGB and depth.
    """
    # Extract colors and points from the point cloud
    colors = np.asarray(pcd.colors)  # RGB values
    points = np.asarray(pcd.points)  # XYZ coordinates
    
    # Compute depth as the Euclidean distance from origin
    depths = np.linalg.norm(points, axis=1)
    
    # Compute histograms for each color channel
    r_hist, _ = np.histogram(colors[:, 0] * 255, bins=num_bins, range=(0, 255))
    g_hist, _ = np.histogram(colors[:, 1] * 255, bins=num_bins, range=(0, 255))
    b_hist, _ = np.histogram(colors[:, 2] * 255, bins=num_bins, range=(0, 255))
    
    # Compute histogram for depth values
    depth_hist, _ = np.histogram(depths, bins=num_bins, range=(np.min(depths), np.max(depths)))
    
    # Normalize histograms
    r_hist = r_hist / r_hist.sum()
    g_hist = g_hist / g_hist.sum()
    b_hist = b_hist / b_hist.sum()
    depth_hist = depth_hist / depth_hist.sum()
    
    # Concatenate all histograms into a single feature vector
    features = np.concatenate([r_hist, g_hist, b_hist, depth_hist])
    
    return features

# Load the point cloud from the specified path
pcd_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

# Check if the point cloud contains color information
if not pcd.has_colors():
    raise ValueError("The point cloud does not contain color information.")

# Extract RGB-D histogram features from the point cloud
features = compute_rgbd_histograms_from_point_cloud(pcd)

# Display the extracted features
print("Extracted RGB-D histogram features:")
print(features)

# Visualize the histograms (optional)
"""

plt.figure(figsize=(10, 5))
plt.title("RGB-D Histogram Features")
plt.bar(range(len(features)), features)
plt.xlabel("Feature Index")
plt.ylabel("Normalized Value")
plt.show()
"""