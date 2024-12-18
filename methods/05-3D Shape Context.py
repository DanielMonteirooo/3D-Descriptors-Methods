import numpy as np
import open3d as o3d

def compute_3d_shape_context(point_cloud, bins_radial=5, bins_theta=12, bins_phi=12, r_min=0.1, r_max=1.0):
    """
    Computes the 3D Shape Context descriptor for each point in the point cloud.

    Parameters:
    - point_cloud (open3d.geometry.PointCloud): Input point cloud.
    - bins_radial (int): Number of bins in the radial direction.
    - bins_theta (int): Number of bins in the azimuthal angle (theta).
    - bins_phi (int): Number of bins in the polar angle (phi).
    - r_min (float): Minimum radius for the spherical shell.
    - r_max (float): Maximum radius for the spherical shell.

    Returns:
    - descriptors (np.ndarray): Array of shape (N, bins_radial * bins_theta * bins_phi) containing
      the 3D Shape Context descriptors for each point.
    """
    points = np.asarray(point_cloud.points)
    N = points.shape[0]
    descriptors = np.zeros((N, bins_radial * bins_theta * bins_phi))

    # Compute pairwise distances and differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    azimuths = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # theta
    elevations = np.arccos(diff[:, :, 2] / (distances + 1e-10))  # phi

    # Define bin edges
    radial_edges = np.logspace(np.log10(r_min), np.log10(r_max), bins_radial + 1)
    theta_edges = np.linspace(-np.pi, np.pi, bins_theta + 1)
    phi_edges = np.linspace(0, np.pi, bins_phi + 1)

    # Compute histograms for each point
    for i in range(N):
        # Exclude the point itself from its neighborhood
        mask = (distances[i] > 1e-10) & (distances[i] <= r_max)
        r_indices = np.digitize(distances[i][mask], radial_edges) - 1
        theta_indices = np.digitize(azimuths[i][mask], theta_edges) - 1
        phi_indices = np.digitize(elevations[i][mask], phi_edges) - 1

        # Accumulate histogram
        for r, t, p in zip(r_indices, theta_indices, phi_indices):
            if 0 <= r < bins_radial and 0 <= t < bins_theta and 0 <= p < bins_phi:
                idx = r * (bins_theta * bins_phi) + t * bins_phi + p
                descriptors[i, idx] += 1

        # Normalize histogram
        descriptors[i] /= np.sum(descriptors[i])

    return descriptors

#Example usage
# Load a point cloud
pcd = o3d.io.read_point_cloud("/Estudos/PIBIC/models/ricardo9/ply/frame0000.ply")

# Compute 3D Shape Context descriptors
descriptors = compute_3d_shape_context(pcd)

# Display the descriptors for the first point
print("Descriptor for the first point:", descriptors[0])
