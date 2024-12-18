import numpy as np
import open3d as o3d

# Define the function to compute Salient Spin Images
def salient_spin_images(point_cloud, radius=0.1, bin_size=0.01, salient_points=None):
    """
    Compute Salient Spin Images for a given point cloud.

    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        radius (float): Radius for the neighborhood.
        bin_size (float): Bin size for spin image discretization.
        salient_points (numpy.ndarray): Optional. Precomputed salient points as an Nx3 numpy array.

    Returns:
        numpy.ndarray: Spin images as an array of size (M, K),
                       where M is the number of salient points, and K is the descriptor length.
    """
    if salient_points is None:
        # Automatically select salient points (e.g., by down-sampling the point cloud)
        salient_points = np.asarray(point_cloud.voxel_down_sample(voxel_size=radius).points)

    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    spin_images = []

    for salient_point in salient_points:
        # Find neighbors within the radius
        [_, idx, _] = kd_tree.search_radius_vector_3d(salient_point, radius)
        neighbors = np.asarray(point_cloud.points)[idx, :]

        # Center neighbors relative to the salient point
        relative_coords = neighbors - salient_point

        # Compute alpha and beta coordinates
        normals = np.asarray(point_cloud.normals)
        salient_normal = normals[idx[0]]
        alpha = np.dot(relative_coords, salient_normal)
        beta = np.linalg.norm(relative_coords - np.outer(alpha, salient_normal), axis=1)

        # Create a histogram (spin image)
        max_bins_alpha = int(2 * radius / bin_size)
        max_bins_beta = int(radius / bin_size)
        spin_image, _, _ = np.histogram2d(
            alpha, beta, bins=[max_bins_alpha, max_bins_beta],
            range=[[-radius, radius], [0, radius]]
        )

        # Flatten the spin image and add to results
        spin_images.append(spin_image.flatten())

    return np.array(spin_images)

# Example usage
if __name__ == "__main__":
    # Load a sample point cloud
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/models/ricardo9/ply/frame0000.ply")

    # Estimate normals if not already present
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute salient spin images
    descriptors = salient_spin_images(pcd, radius=0.1, bin_size=0.01)

    # Print the shape and a snippet of the descriptors
    print("Spin Images Shape:", descriptors.shape)
    print("Sample Descriptor:", descriptors[0])
