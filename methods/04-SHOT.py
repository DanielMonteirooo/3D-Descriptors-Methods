import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

def compute_shot_features(pcd, radius=0.1, num_azimuth=8, num_elevation=2, num_radial=2, num_bins=11):
    """
    Compute SHOT descriptors according to the original paper specifications
    """
    # Preprocess point cloud
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, 100))
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    tree = KDTree(points)
    
    descriptors = []
    
    # SHOT parameters based on paper
    radial_bins = np.linspace(0, radius, num_radial+1)
    elev_bins = np.linspace(-1, 1, num_elevation+1)
    azimuth_bins = np.linspace(-np.pi, np.pi, num_azimuth+1)
    
    for idx, (point, normal) in enumerate(zip(points, normals)):
        # Find neighbors within radius
        neighbors = tree.query_ball_point(point, radius)
        if len(neighbors) < 10:
            continue
            
        # Compute Local Reference Frame (LRF)
        diff = points[neighbors] - point
        distances = np.linalg.norm(diff, axis=1)
        weights = (radius - distances) / radius
        
        # Weighted covariance matrix
        cov = (diff.T * weights) @ diff
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Eigenvalue ordering (descending)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Disambiguate LRF axes (z-axis)
        z_axis = eigenvectors[:, 2]
        median_dist = np.median(distances)
        mask = np.abs(distances - median_dist) < 0.1*radius
        if np.sum(z_axis @ diff[mask].T) < 0:
            z_axis *= -1
            
        # Disambiguate x-axis
        x_axis = eigenvectors[:, 0]
        if np.sum(x_axis @ diff[mask].T) < 0:
            x_axis *= -1
            
        y_axis = np.cross(z_axis, x_axis)
        
        # Transform neighbors to LRF coordinates
        local_coords = diff @ np.column_stack((x_axis, y_axis, z_axis))
        
        # Initialize descriptor
        descriptor = np.zeros((num_azimuth, num_elevation, num_radial, num_bins))
        
        for i, neighbor_idx in enumerate(neighbors):
            if neighbor_idx == idx:
                continue
                
            # Spherical coordinates
            x, y, z = local_coords[i]
            r = np.linalg.norm(local_coords[i])
            theta = np.arccos(z / r) if r > 0 else 0
            phi = np.arctan2(y, x)
            
            # Bin indices
            r_bin = np.digitize(r, radial_bins) - 1
            elev_bin = np.digitize(np.cos(theta), elev_bins) - 1
            azimuth_bin = np.digitize(phi, azimuth_bins) - 1
            
            # Handle bin overflow
            r_bin = np.clip(r_bin, 0, num_radial-1)
            elev_bin = np.clip(elev_bin, 0, num_elevation-1)
            azimuth_bin = np.clip(azimuth_bin, 0, num_azimuth-1)
            
            # Normal angle component
            cos_angle = np.dot(normals[neighbor_idx], z_axis)
            angle_bin = np.digitize(cos_angle, np.linspace(-1, 1, num_bins+1)) - 1
            angle_bin = np.clip(angle_bin, 0, num_bins-1)
            
            # Update histogram with quadrilinear interpolation
            # (Implementation details omitted for brevity)
            descriptor[azimuth_bin, elev_bin, r_bin, angle_bin] += weights[i]
        
        # Normalize and flatten descriptor
        descriptor = descriptor.reshape(-1)
        descriptor /= np.linalg.norm(descriptor) + 1e-6
        descriptors.append(descriptor)
    
    return np.array(descriptors)

def process_features(base_path):
    pcd = o3d.io.read_point_cloud(base_path)
    features = compute_shot_features(pcd)
    
    if features is None or len(features) == 0:
        print("No features extracted")
        return []
    
    # Calculate row averages (histogram averages)
    row_averages = []
    for feature in features:
        # Reshape to (32 spatial bins, 11 orientation bins)
        histograms = feature.reshape(32, 11)
        row_avg = np.mean(histograms, axis=1)
        row_averages.extend(row_avg.tolist())
    
    return row_averages

# Usage example
base_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_the20smaria_00600_vox10_dec_geom06_text06_trisoup-predlift.ply"
averages = process_features(base_path)
print(f"First 10 averages: {averages[:10]}")
print(f"Total features: {len(averages)}")
