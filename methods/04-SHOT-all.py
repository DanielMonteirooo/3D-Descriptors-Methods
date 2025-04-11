import open3d as o3d
import numpy as np
import pandas as pd
import os

def compute_shot_features(point_cloud, radius):
    """
    Compute SHOT (Signature of Histograms of Orientations) descriptors for a point cloud.
    """
    if not point_cloud.has_normals():
        print("Computing normals for the point cloud...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    shot_descriptors = []
    
    for i in range(len(point_cloud.points)):
        _, idx, _ = kdtree.search_radius_vector_3d(point_cloud.points[i], radius)
        if len(idx) < 5:
            shot_descriptors.append(np.zeros(352))
            continue
        
        neighbors = np.asarray(point_cloud.points)[idx, :]
        normals = np.asarray(point_cloud.normals)[idx, :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]
        
        reference_normal = np.asarray(point_cloud.normals)[i]
        cos_theta = np.dot(normals, reference_normal)
        hist, _ = np.histogram(cos_theta, bins=11, range=(-1.0, 1.0))
        descriptor = hist / np.linalg.norm(hist)
        shot_descriptors.append(descriptor)
    
    return np.array(shot_descriptors)

def calculate_feature_averages(features):
    """
    Calculate the average of row averages and column averages of a feature matrix.
    """
    row_averages = np.mean(features, axis=1)
    column_averages = np.mean(features, axis=0)
    overall_average = (np.mean(row_averages) + np.mean(column_averages)) / 2.0
    return overall_average

def process_csv_and_compute_features(csv_path, base_path, radius=0.05):
    """
    Process the CSV file and compute SHOT features for each SIGNAL in the file.
    """
    data = pd.read_csv(csv_path)
    signal_column = data['SIGNAL']
    
    for signal in signal_column:
        file_path = os.path.join(base_path, signal.strip())
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        print(f"Processing: {file_path}")
        
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            print(f"Failed to load point cloud: {file_path}")
            continue
        
        shot_features = compute_shot_features(pcd, radius)
        average_value = calculate_feature_averages(shot_features)
        
        # Print result immediately after processing
        print(f"Signal: {signal}, Average: {average_value}")

if __name__ == "__main__":
    csv_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv"
    base_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/"
    
    print("Processing CSV and computing SHOT features...")
    process_csv_and_compute_features(csv_path, base_path)
