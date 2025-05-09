import numpy as np
import open3d as o3d
from scipy.special import sph_harm

def spherical_harmonic_descriptors(point_cloud, l_max=4):
    """
    Compute Spherical Harmonic Descriptors (SHD) for a given point cloud.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input 3D point cloud.
        l_max (int): Maximum degree of spherical harmonics.

    Returns:
        np.ndarray: Spherical Harmonic Descriptors.
    """
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)

    # Convert Cartesian coordinates to spherical coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-8))  # Avoid division by zero
    phi = np.arctan2(y, x)

    # Compute spherical harmonics coefficients
    sh_coefficients = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute the spherical harmonics
            Y_lm = sph_harm(m, l, phi, theta)
            # Project the function onto the spherical harmonic basis
            projection = np.sum(r * Y_lm)  # Weighted by radius
            sh_coefficients.append(np.abs(projection))  # Use magnitude of projection

    return np.array(sh_coefficients)

# Example usage:
if __name__ == "__main__":
    # Load a sample point cloud
    pcd = o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)

    # Downsample the point cloud to reduce computation
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Extract Spherical Harmonic Descriptors
    shd_features = spherical_harmonic_descriptors(voxel_down_pcd, l_max=4)

    # Display the features
    print("Extracted Spherical Harmonic Descriptors:")
    print(shd_features)

# Reference: https://github.com/marstaa/PySphereX Last access: 18/12/2024
# Reference: https://github.com/tsutterley/model-harmonics Last access: 18/12/2024