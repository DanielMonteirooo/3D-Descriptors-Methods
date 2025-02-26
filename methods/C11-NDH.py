import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_ndh(point_cloud, bins=32):
    """
    Computes the RGB Normalized Difference Histograms (NDH) for a given point cloud.

    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud with RGB colors.
        bins (int): Number of bins for the histogram.

    Returns:
        dict: A dictionary containing histograms for R-G, R-B, and G-B normalized differences.
    """
    # Extract RGB colors from the point cloud
    colors = np.asarray(point_cloud.colors)
    if colors.shape[0] == 0:
        raise ValueError("Point cloud does not have color information.")

    # Compute normalized differences
    r_minus_g = (colors[:, 0] - colors[:, 1]) / (colors[:, 0] + colors[:, 1] + 1e-6)
    r_minus_b = (colors[:, 0] - colors[:, 2]) / (colors[:, 0] + colors[:, 2] + 1e-6)
    g_minus_b = (colors[:, 1] - colors[:, 2]) / (colors[:, 1] + colors[:, 2] + 1e-6)

    # Compute histograms
    hist_r_g, _ = np.histogram(r_minus_g, bins=bins, range=(-1, 1), density=True)
    hist_r_b, _ = np.histogram(r_minus_b, bins=bins, range=(-1, 1), density=True)
    hist_g_b, _ = np.histogram(g_minus_b, bins=bins, range=(-1, 1), density=True)

    return {
        "R-G": hist_r_g,
        "R-B": hist_r_b,
        "G-B": hist_g_b
    }

# Specify the path to your point cloud file
point_cloud_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the point cloud
pcd = o3d.io.read_point_cloud(point_cloud_path)

# Check if the point cloud is loaded successfully
if not pcd.has_colors():
    raise ValueError("The loaded point cloud does not have color information.")

# Compute NDH features
ndh_features = compute_ndh(pcd)

# Print NDH results to console
print("Normalized Difference Histograms (NDH):")
for key, hist in ndh_features.items():
    print(f"{key} Histogram:")
    print(hist)
    print()

# Display the extracted features using bar plots
'''

plt.figure(figsize=(12, 4))
for i, (key, hist) in enumerate(ndh_features.items()):
    plt.subplot(1, 3, i + 1)
    plt.bar(np.linspace(-1, 1, len(hist)), hist, width=0.05)
    plt.title(f"{key} Histogram")
    plt.xlabel("Normalized Difference")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
'''
