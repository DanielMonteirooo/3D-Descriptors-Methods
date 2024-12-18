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
    point_cloud_path = "/home/dani/Estudos/PIBIC/models/frame0000.pcd"

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # Extract FPFH features
    pcd_down, fpfh = extract_fpfh_features(pcd)

    # Display the downsampled point cloud
    o3d.visualization.draw_geometries([pcd_down], window_name="Downsampled Point Cloud")

    # Display the FPFH features
    print("FPFH feature dimensions:", fpfh.dimension())
    print("Number of FPFH features:", fpfh.num())
    print("FPFH features data:")
    print(fpfh.data)
