import open3d as o3d
import numpy as np

def color_vfh(point_cloud):
    """
    Implements the Color-VFH (C-VFH) descriptor for a 3D colored point cloud.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input 3D colored point cloud.

    Returns:
        np.ndarray: Extracted Color-VFH features.
    """
    # Compute normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Extract RGB values and normalize them
    colors = np.asarray(point_cloud.colors)
    normalized_colors = colors.mean(axis=0)  # Mean RGB values
    normalized_colors /= 255.0  # Normalize to [0, 1]

    # Compute the FPFH descriptor for shape information
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Combine shape (FPFH) and color information
    color_vfh_features = np.hstack((fpfh.data.flatten(), normalized_colors))

    return color_vfh_features

# Path to the specified 3D colored point cloud file
file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# Load the 3D colored point cloud from the specified path
point_cloud = o3d.io.read_point_cloud(file_path)

# Check if the point cloud is loaded successfully
if not point_cloud.is_empty():
    print("Point cloud loaded successfully!")

    # Extract Color-VFH features
    features = color_vfh(point_cloud)

    # Display the extracted features
    print("Extracted Color-VFH Features:")
    print(features)
else:
    print("Failed to load point cloud. Please check the file path.")
