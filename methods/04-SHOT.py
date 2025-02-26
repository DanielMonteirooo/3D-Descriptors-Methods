# Need to install pyshot package, which is a Python wrapper for the SHOT descriptor implementation in C++. 
# More information can be found at:https://github.com/uhlmanngroup/pyshot

import numpy as np
import open3d as o3d
from pyshot import get_descriptors

def compute_shot_features(point_cloud, radius=0.1, min_neighbors=3, n_bins=20):
    # Ensure the point cloud has normals
    if not point_cloud.has_normals():
        raise ValueError("Point cloud must have normals computed.")

    # Convert Open3D point cloud to numpy arrays
    vertices = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    # Compute SHOT descriptors
    descriptors = get_descriptors(
        vertices=vertices,
        faces=None,  # Faces are optional; can be None if not available
        radius=radius,
        local_rf_radius=radius,
        min_neighbors=min_neighbors,
        n_bins=n_bins,
        double_volumes_sectors=True,
        use_interpolation=True,
        use_normalization=True,
    )

    return descriptors

# Load point cloud using Open3D
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")

# Compute normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Compute SHOT features
shot_features = compute_shot_features(pcd)

# Display the shape of the extracted features
print("Extracted SHOT features shape:", shot_features.shape)

#Reference: https://github.com/uhlmanngroup/pyshot? Last access: 18/12/2024
