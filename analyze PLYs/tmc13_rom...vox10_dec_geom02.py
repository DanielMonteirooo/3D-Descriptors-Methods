#Here's how to analyze your PLY file and feature vectors:

# Import necessary libraries

import numpy as np
from plyfile import PlyData

def analyze_point_cloud(ply_path):
    """Analyze PLY file structure and feature dimensions"""
    try:
        # Read PLY file
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        
        # Get point count and properties
        num_points = vertices.count
        properties = vertices.data.dtype.names
        prop_count = len(properties)
        
        print(f"PLY File Analysis:")
        print(f"Points in dataset: {num_points:,}")
        print(f"Properties per point: {prop_count}")
        print(f"Property names: {properties}")
        
        return num_points, prop_count
        
    except Exception as e:
        print(f"Error reading PLY file: {str(e)}")
        return None, None

def analyze_features(feature_array):
    """Analyze computed feature vectors"""
    if feature_array is None:
        return
        
    print("\nFeature Vector Analysis:")
    print(f"Total points processed: {feature_array.shape[0]:,}")
    print(f"Features per point: {feature_array.shape[1]}")
    print(f"Feature vector structure: {feature_array.dtype}")
    print(f"Memory usage: {feature_array.nbytes/1024:.2f} KB")

# Usage
base_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/"
ply_file = "tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

# 1. Analyze original PLY file
num_pts, num_props = analyze_point_cloud(base_path + ply_file)

# 2. Compute features (using previous implementation)
point_cloud = PlyData.read(base_path + ply_file)['vertex'].data
xyz = np.vstack([point_cloud[i] for i in ['x', 'y', 'z']]).T
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.cpu.pybind.geometry import KDTreeSearchParamHybrid

# Convert xyz para Open3D PointCloud
pcd = PointCloud()
pcd.points = Vector3dVector(xyz)

# Calcular normais
pcd.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

# Compute FPFH features
search_param = KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
fpfh_features = compute_fpfh_feature(pcd, search_param)

# Converta o objeto Feature para um array NumPy
fpfh_features_np = np.asarray(fpfh_features.data).T

# 3. Analyze feature vectors
analyze_features(fpfh_features_np)

