import open3d as o3d
import numpy as np

def compute_vfh(point_cloud):
    """
    Compute the Viewpoint Feature Histogram (VFH) for the given point cloud.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud for feature extraction.
    
    Returns:
        np.ndarray: The extracted VFH features.
    """
    # Ensure the point cloud has normals computed
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    
    # Compute the VFH descriptor
    vfh = o3d.pipelines.registration.Feature()
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100)
    
    vfh.compute(point_cloud, search_param)
    
    # Return the VFH feature matrix as a numpy array
    return np.array(vfh.data).T

# Example usage
if __name__ == "__main__":
    # Load a sample point cloud
    point_cloud = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")
    
    if point_cloud.is_empty():
        raise ValueError("Loaded point cloud is empty. Please provide a valid point cloud file.")
    
    # Compute VFH features
    vfh_features = compute_vfh(point_cloud)
    
    # Display the computed features
    print("Extracted VFH Features:")
    print(vfh_features)

    # Optional: Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
