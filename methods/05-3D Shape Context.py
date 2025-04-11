import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def compute_3d_shape_context(pcd, num_radial_bins=5, num_azimuth_bins=12, 
                            num_polar_bins=4, radius=0.1):
    """
    Compute 3D Shape Context descriptors for a point cloud
    """
    points = np.asarray(pcd.points)
    kdtree = cKDTree(points)
    
    # Parameters for spherical bins
    radial_edges = np.linspace(0, radius, num_radial_bins+1)
    azimuth_edges = np.linspace(0, 2*np.pi, num_azimuth_bins+1)
    polar_edges = np.linspace(0, np.pi, num_polar_bins+1)
    
    descriptors = []
    for i, point in enumerate(points):
        # Find neighbors within radius
        indices = kdtree.query_ball_point(point, radius)
        neighbors = points[indices] - point
        
        # Convert to spherical coordinates
        r = np.linalg.norm(neighbors, axis=1)
        theta = np.arctan2(neighbors[:,1], neighbors[:,0]) + np.pi  # Azimuth [0, 2π]
        phi = np.arccos(neighbors[:,2] / (r + 1e-8))                # Polar [0, π]
        
        # Create 3D histogram
        hist, _ = np.histogramdd(
            np.column_stack((r, theta, phi)),
            bins=(radial_edges, azimuth_edges, polar_edges)
        )
        
        # Normalize and flatten histogram
        hist = hist / (len(neighbors) + 1e-8)
        descriptors.append(hist.flatten())
    
    return np.array(descriptors)

def visualize_features(pcd, descriptors):
    """Visualize point cloud with feature intensity coloring"""
    colors = np.zeros_like(pcd.points)
    
    # Use mean descriptor value for coloring
    colors[:,0] = descriptors.mean(axis=1)  # Red channel
    colors[:,1] = descriptors.max(axis=1)    # Green channel
    colors[:,2] = descriptors.min(axis=1)    # Blue channel
    
    # Normalize colors
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# Example usage
if __name__ == "__main__":
    # Load sample data
    pcd1 = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")


    # Compute descriptors
    descriptors = compute_3d_shape_context(pcd1)
    
    # Visualize features
    # visualize_features(pcd1, descriptors)
    
    # Calculate averages
    row_avg = descriptors.mean(axis=1).mean()
    col_avg = descriptors.mean(axis=0).mean()
    final_avg = (row_avg + col_avg) / 2
    
    print(f"Row average: {row_avg:.4f}")
    print(f"Column average: {col_avg:.4f}")
    print(f"Final combined average: {final_avg:.4f}")
