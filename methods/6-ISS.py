import open3d as o3d
import time

def compute_iss_keypoints(point_cloud, salient_radius, non_max_radius, gamma_21, gamma_32):
    """
    Compute ISS keypoints for a given point cloud.

    Parameters:
    - point_cloud (o3d.geometry.PointCloud): The input point cloud.
    - salient_radius (float): The radius for computing the salient points.
    - non_max_radius (float): The radius for non-maximum suppression.
    - gamma_21 (float): The threshold for the ratio between the second and first eigenvalues.
    - gamma_32 (float): The threshold for the ratio between the third and second eigenvalues.

    Returns:
    - o3d.geometry.PointCloud: The point cloud containing the ISS keypoints.
    """
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        point_cloud,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=gamma_21,
        gamma_32=gamma_32
    )
    return keypoints

# Example usage:
if __name__ == "__main__":
    # Load a sample point cloud
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_poisson_disk(5000)

    # Define ISS parameters
    salient_radius = 0.005
    non_max_radius = 0.005
    gamma_21 = 0.5
    gamma_32 = 0.5

    # Compute ISS keypoints
    start_time = time.time()
    iss_keypoints = compute_iss_keypoints(point_cloud, salient_radius, non_max_radius, gamma_21, gamma_32)
    end_time = time.time()
    print(f"ISS keypoints computation took {end_time - start_time:.2f} seconds.")

    # Visualize the keypoints
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    iss_keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([point_cloud, iss_keypoints], point_show_normal=False)
