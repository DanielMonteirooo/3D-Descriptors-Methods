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
    
    # Calculate and print row averages
    row_averages = descriptors.mean(axis=1)
    print("## Per-Row Averages ##")
    for idx, avg in enumerate(row_averages):
        print(f"Row {idx:04d}: {avg:.6f}")
    
    # Generate and print all rows
    print("\n## All Rows in Sequence ##")
    for idx, row in enumerate(descriptors):
        print(f"\nRow {idx:04d} features:")
        print(np.array2string(row, precision=4, suppress_small=True, separator=', '))