import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_fpfh(point_cloud, radius, n_bins=11):
    """
    Compute Fast Point Feature Histograms (FPFH) with angular feature relationships.
    
    Parameters:
        point_cloud (np.ndarray): Nx3 array of 3D points
        radius (float): Neighborhood search radius
        n_bins (int): Number of histogram bins per feature
        
    Returns:
        np.ndarray: FPFH features (N x 33)
    """
    # 1. Compute normals using PCA (simplified version)
    nbrs = NearestNeighbors(n_neighbors=10).fit(point_cloud)
    _, indices = nbrs.kneighbors(point_cloud)
    cov_mats = [np.cov(point_cloud[indices[i]].T) for i in range(len(point_cloud))]
    normals = np.array([np.linalg.eigh(c)[1][:,0] for c in cov_mats])
    
    # 2. Compute SPFH features
    hist_bins = [np.linspace(0, np.pi, n_bins+1) for _ in range(3)]
    spfh = np.zeros((len(point_cloud), 3*n_bins))
    
    nbrs_radius = NearestNeighbors(radius=radius).fit(point_cloud)
    
    for i, (pt, normal) in enumerate(zip(point_cloud, normals)):
        neighbors = nbrs_radius.radius_neighbors([pt], return_distance=False)[0]
        neighbors = np.setdiff1d(neighbors, [i])  # Exclude self
        
        if len(neighbors) < 2:
            continue
            
        features = []
        for j in neighbors:
            delta = point_cloud[j] - pt
            u = normal
            v = np.cross(delta, u)
            w = np.cross(u, v)
            
            # Compute angular features
            alpha = np.arctan2(np.dot(w, normals[j]), np.dot(u, normals[j]))
            beta = np.linalg.norm(v) / np.linalg.norm(delta)
            theta = np.arccos(np.clip(np.dot(delta/np.linalg.norm(delta), u), -1, 1))
            
            features.append([alpha, beta, theta])
        
        # Generate triple histogram
        hist = np.concatenate([
            np.histogram(np.array(features)[:,0], bins=hist_bins[0])[0],
            np.histogram(np.array(features)[:,1], bins=hist_bins[1])[0],
            np.histogram(np.array(features)[:,2], bins=hist_bins[2])[0]
        ])
        spfh[i] = hist / (hist.sum() + 1e-6)
    
    # 3. Compute FPFH with weighted neighbors
    fpfh = np.zeros_like(spfh)
    for i, pt in enumerate(point_cloud):
        neighbors = nbrs_radius.radius_neighbors([pt], return_distance=False)[0]
        if len(neighbors) == 0:
            fpfh[i] = spfh[i]
            continue
            
        # Calculate distance weights
        dists = np.linalg.norm(point_cloud[neighbors] - pt, axis=1)
        weights = 1 / (dists + 1e-6)
        
        # Combine features
        fpfh[i] = spfh[i] + (weights @ spfh[neighbors]) / (weights.sum() + 1e-6)
    
    return np.nan_to_num(fpfh, nan=0.0)

def print_features(features, num_samples=5):
    """Print feature vectors and statistics in a human-readable format"""
    print("\n" + "="*50)
    print(f"FPFH Feature Matrix Shape: {features.shape}")
    print(f"First {num_samples} feature vectors:")
    for i in range(num_samples):
        print(f"Point {i+1}:")
        print(np.round(features[i], 3))

    # Calculate and print statistics
    row_means = features.mean(axis=1)
    print("\n" + "-"*50)
    print("Row averages (first 50 points):")
    print(np.round(row_means[:50], 3))
    
    print("\n" + "-"*50)
    print(f"Global statistics:\n"
          f"Min average: {row_means.min():.3f}\n"
          f"Max average: {row_means.max():.3f}\n"
          f"Mean average: {row_means.mean():.3f}")

def process_and_display(point_cloud, radius):
    """Full processing pipeline with visualization"""
    # Compute features
    fpfh_features = compute_fpfh(point_cloud, radius)
    
    # Calculate and format row averages
    row_averages = fpfh_features.mean(axis=1).tolist()
    formatted_averages = [f"{x:.4f}" for x in row_averages]
    
    # Display results
    print_features(fpfh_features)
    
    print("\n" + "="*50)
    print("Complete list of row averages:")
    print("[" + ",\n".join(formatted_averages) + "]")
    
    return fpfh_features


# Example usage
if __name__ == "__main__":
    base_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"
    
    # Generate or load your point cloud (example with random data)
    example_cloud = np.random.rand(100, 3)  # Alterar para carregar o PLY
    radius = 0.5
    
    # Run processing pipeline
    features = process_and_display(example_cloud, radius)
    
    # Save features
    np.save(f"{base_path}fpfh_features.npy", features)