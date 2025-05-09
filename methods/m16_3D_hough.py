import numpy as np
import open3d as o3d

def compute_3d_hough_transform(point_cloud, voxel_size=0.05):
    """
    Perform 3D Hough Transform on a point cloud to detect objects.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input 3D point cloud.
        voxel_size (float): Voxel size for downsampling the point cloud.
    
    Returns:
        keypoints (o3d.geometry.PointCloud): Keypoints detected in the point cloud.
    """
    # Downsample the point cloud for faster processing
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)
    
    # Estimate normals for the downsampled point cloud
    downsampled_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Compute ISS Keypoints using Open3D
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        downsampled_pcd,
        salient_radius=voxel_size * 2,
        non_max_radius=voxel_size * 2,
        gamma_21=0.5,
        gamma_32=0.5
    )
    
    return keypoints

def visualize_features(original_pcd, keypoints):
    """
    Visualize the original point cloud with extracted keypoints.
    
    Parameters:
        original_pcd (o3d.geometry.PointCloud): Original input point cloud.
        keypoints (o3d.geometry.PointCloud): Extracted keypoints.
    """
    # Colorize the original point cloud and keypoints for visualization
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for original
    keypoints.paint_uniform_color([1.0, 0.75, 0.0])   # Yellow for keypoints
    
    # Visualize both together
    o3d.visualization.draw_geometries([original_pcd, keypoints])

# Main function to demonstrate usage
if __name__ == "__main__":
    # Path to your PLY file
    ply_file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
    
    # Load the point cloud from the provided path
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    if not pcd.has_points():
        print("Error: The loaded point cloud is empty or invalid.")
        exit(1)
    
    print("Loaded point cloud successfully.")
    
    # Perform 3D Hough Transform to extract features
    extracted_keypoints = compute_3d_hough_transform(pcd)
    
    print(f"Extracted {len(extracted_keypoints.points)} keypoints from the point cloud.")
    
    # Visualize the results
    #visualize_features(pcd, extracted_keypoints)
