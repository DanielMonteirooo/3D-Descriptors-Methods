"""
import open3d as o3d

def extract_fpfh_features(pcd, voxel_size=0.05):
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud to {len(pcd_down.points)} points.")

    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    print("Estimated normals.")

    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
    print("Computed FPFH features.")

    return pcd_down, fpfh

if __name__ == "__main__":
    # Path to your point cloud file
    point_cloud_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
    #point_cloud_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/references/romanoillamp_vox10.ply" #Arquivo pristino

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Extract FPFH features
    pcd_down, fpfh = extract_fpfh_features(pcd)

    # Display the downsampled point cloud
    #o3d.visualization.draw_geometries([pcd_down], window_name="Downsampled Point Cloud")

    # Display the FPFH features
    print("FPFH feature dimensions:", fpfh.dimension())
    print("Number of FPFH features:", fpfh.num())
    print("FPFH features data:")
    print(fpfh.data.mean())
"""

import open3d as o3d
import numpy as np

def extract_fpfh_features(pcd, voxel_size=0.05):
    """Extract FPFH features from a point cloud using Open3D"""
    # Downsample and process point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    
    # Estimate normals
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100
        )
    )
    return pcd_down, fpfh

def compute_feature_statistics(fpfh):
    """Calculate row/column averages and combined average"""
    feature_matrix = fpfh.data
    
    # Row averages (per-feature-dimension averages)
    row_means = np.mean(feature_matrix, axis=1)
    avg_row_means = np.mean(row_means)
    
    # Column averages (per-point feature averages)
    col_means = np.mean(feature_matrix, axis=0)
    avg_col_means = np.mean(col_means)
    
    # Combined final average
    final_avg = (avg_row_means + avg_col_means) / 2
    return avg_row_means, avg_col_means, final_avg

if __name__ == "__main__":
    # Load point cloud
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")
    
    # Feature extraction
    pcd_down, fpfh_features = extract_fpfh_features(pcd)
    
    # Display feature information
    print("\nFPFH Feature Matrix Properties:")
    print(f"Feature dimensions: {fpfh_features.dimension()}")
    print(f"Number of features: {fpfh_features.num()}")
    print(f"Matrix shape: {fpfh_features.data.shape}")
    
    # Calculate statistics
    row_avg, col_avg, combined_avg = compute_feature_statistics(fpfh_features)
    
    print("\nStatistical Analysis:")
    print(f"Average of row means: {row_avg:.6f}")
    print(f"Average of column means: {col_avg:.6f}")
    print(f"Combined average: {combined_avg:.6f}")
