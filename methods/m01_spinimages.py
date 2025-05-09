import open3d as o3d
import numpy as np

def compute_spin_image(point_cloud, query_point_index, bin_size=0.01, image_width=10):
    if not point_cloud.has_normals():
        raise ValueError("Point cloud must have computed normals")
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    query_point = np.asarray(points[query_point_index]).reshape(-1)
    query_normal = np.asarray(normals[query_point_index]).reshape(-1)
    # Garante dimensão correta
    if query_point.shape[0] != 3 or query_normal.shape[0] != 3:
        raise ValueError("Ponto ou normal não tem dimensão 3")
    spin_image = np.zeros((image_width, image_width))
    for point in points:
        point = np.asarray(point).reshape(-1)
        if point.shape[0] != 3:
            continue  # ignora pontos inválidos
        vector = point - query_point
        if vector.shape[0] != 3:
            continue
        alpha = np.dot(vector, query_normal)
        beta = np.linalg.norm(np.cross(vector, query_normal))
        i = int(alpha/bin_size + image_width//2)
        j = int(beta/bin_size)
        if 0 <= i < image_width and 0 <= j < image_width:
            spin_image[j, i] += 1
    return spin_image

def extract_spinimages(pcd, num_samples=50, bin_size=0.01, image_width=10):
    n_points = np.asarray(pcd.points).shape[0]
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    if n_points < num_samples:
        indices = np.arange(n_points)
    else:
        indices = np.linspace(0, n_points-1, num=num_samples, dtype=int)
    spin_vectors = []
    for idx in indices:
        spin_img = compute_spin_image(pcd, idx, bin_size, image_width)
        spin_vectors.append(spin_img.flatten())
    global_vector = np.mean(np.stack(spin_vectors), axis=0)
    return global_vector
