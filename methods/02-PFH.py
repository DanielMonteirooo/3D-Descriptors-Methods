import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty or file could not be read.")
    return pcd

def estimate_normals(pcd, radius=1.0, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()

def compute_fpfh_open3d(pcd, radius=1.5, max_nn=100):
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.array(fpfh.data).T  # (N, 33)

def print_features(features, num_samples=5):
    print("\n" + "="*50)
    print(f"FPFH Feature Matrix Shape: {features.shape}")
    print(f"First {num_samples} feature vectors:")
    for i in range(num_samples):
        print(f"Point {i+1}:", np.round(features[i], 3))
    row_means = features.mean(axis=0)
    print("\n" + "-"*50)
    print("Row averages (first 50 points):")
    print(np.round(row_means[:50], 3))
    print("\n" + "-"*50)
    print(f"Global statistics:\n"
          f"Min average: {row_means.min():.3f}\n"
          f"Max average: {row_means.max():.3f}\n"
          f"Mean average: {row_means.mean():.3f}")


if __name__ == "__main__":
    base_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_loot_vox10_1200_dec_geom01_text01_octree-predlift.ply"
    pcd = load_point_cloud(base_path)
    estimate_normals(pcd, radius=1.0, max_nn=30)
    features = compute_fpfh_open3d(pcd, radius=1.5, max_nn=100)
    print_features(features)
    np.save(f"{base_path}_fpfh_features.npy", features)