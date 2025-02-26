import open3d as o3d
import numpy as np
from sklearn.preprocessing import normalize
import colorsys  # For RGB to HSV conversion

def rgb_to_hsv(rgb):
    """
    Convert an RGB color (range 0-1) to HSV.
    Parameters:
        rgb (array-like): RGB values in range [0, 1].
    Returns:
        np.ndarray: HSV values in range [0, 1].
    """
    return np.array(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))

def compute_pfh_color(point_cloud, radius):
    """
    Compute PFH+Color features for a given point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud with colors.
        radius (float): Radius for neighborhood search.

    Returns:
        np.ndarray: Combined PFH+Color features for each point.
    """

    # Ensure the point cloud is in legacy format
    if isinstance(point_cloud, o3d.t.geometry.PointCloud):
        point_cloud = point_cloud.to_legacy_pointcloud()

    # Step 1: Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Step 2: Convert RGB to HSV for color features
    colors = np.asarray(point_cloud.colors)
    hsv_colors = np.apply_along_axis(rgb_to_hsv, 1, colors)

    # Step 3: Compute PFH features
    pfh_features = []
    for i, point in enumerate(point_cloud.points):
        # Find neighbors within the radius
        [_, idx, _] = point_cloud.search_radius_vector_3d(point, radius)
        if len(idx) < 2:
            continue

        # Compute geometric PFH-like feature (pairwise relationships)
        geometric_features = []
        for j in idx[1:]:  # Skip the query point itself
            diff = np.asarray(point_cloud.points[j]) - np.asarray(point)
            geometric_features.append(diff)

        # Combine geometric features with HSV color values of the query point
        combined_features = np.concatenate((np.mean(geometric_features, axis=0), hsv_colors[i]))
        pfh_features.append(combined_features)

    # Normalize features for consistency
    pfh_features = normalize(np.array(pfh_features), axis=0)

    return pfh_features


# Load the point cloud from the specified path
pcd_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

# Display basic information about the loaded point cloud
print("Loaded point cloud:")
print(pcd)

# Compute PFH+Color features
radius = 0.05  # Set neighborhood radius
features = compute_pfh_color(pcd, radius)

# Display the extracted features
print("Extracted PFH+Color features:")
print(features)
