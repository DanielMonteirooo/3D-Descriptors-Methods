#first: pip install open3d



import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_amphoriskos_vox10_dec_geom04_text04_octree-predlift.ply")

# Downsample the point cloud
voxel_size = 0.05
downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# Estimate normals
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=voxel_size * 2, max_nn=30))

# Normalize normals to unit vectors
downpcd.normalize_normals()

# Option 1: Orient normals towards a specific direction
# For example, aligning with the positive Z direction
downpcd.orient_normals_to_align_with_direction([0, 0, 1])

# Option 2: Orient normals towards a camera location
# Specify the camera location; for instance, the origin
downpcd.orient_normals_towards_camera_location([0, 0, 0])

import numpy as np

def compute_hon(point_cloud, bins=20):
    # Extract normals
    normals = np.asarray(point_cloud.normals)
    
    # Convert normals to spherical coordinates
    azimuths = np.arctan2(normals[:, 1], normals[:, 0])
    elevations = np.arcsin(normals[:, 2])
    
    # Normalize angles to [0, 1] range
    azimuths = (azimuths + np.pi) / (2 * np.pi)
    elevations = (elevations + np.pi / 2) / np.pi
    
    # Compute histogram
    hist, edges = np.histogramdd(
        np.stack((azimuths, elevations), axis=1),
        bins=bins,
        range=[[0, 1], [0, 1]]
    )
    
    # Normalize histogram
    hist /= np.sum(hist)
    
    return hist

# Compute HoN descriptor
hon_descriptor = compute_hon(downpcd)


import matplotlib.pyplot as plt

# Plot the HoN descriptor
"""

plt.imshow(hon_descriptor, interpolation='nearest', cmap='viridis')
plt.title('Histogram of Oriented Normals (HoN)')
plt.xlabel('Azimuth Bins')
plt.ylabel('Elevation Bins')
plt.colorbar(label='Frequency')
plt.show()
"""
# Reference: 
# https://github.com/PFMassiani/consistent_normals_orientation? Last access: 18/12/2024
# https://github.com/MeshInspector/MeshLib/wiki/Consistent-orientation-of-normals-in-point-clouds? Last access: 18/12/2024