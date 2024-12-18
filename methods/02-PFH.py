import open3d as o3d
import numpy as np

def compute_pfh(point_cloud, search_radius=0.05, nb_bins=5):
    """
    Compute Point Feature Histograms (PFH) for a given point cloud.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        search_radius (float): Radius used for nearest neighbor search.
        nb_bins (int): Number of bins for each angular feature.
    
    Returns:
        np.ndarray: The computed PFH features as a NumPy array.
    """
    # Ensure the point cloud has normals computed
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=30))

    # Create KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    # Initialize list to hold PFH features
    pfh_features = []

    # Iterate over all points in the point cloud
    for i in range(len(point_cloud.points)):
        # Find neighbors within the search radius
        [k, idx, _] = kdtree.search_radius_vector_3d(
            point_cloud.points[i], search_radius)
        if k < 5:  # Ensure there are enough neighbors
            pfh_features.append(np.zeros(nb_bins**3))
            continue

        # Initialize histogram
        hist = np.zeros((nb_bins, nb_bins, nb_bins))

        # Iterate over all pairs of neighbors
        for j in range(1, k):
            for l in range(j + 1, k):
                p1 = np.asarray(point_cloud.points)[idx[j]]
                p2 = np.asarray(point_cloud.points)[idx[l]]
                n1 = np.asarray(point_cloud.normals)[idx[j]]
                n2 = np.asarray(point_cloud.normals)[idx[l]]

                # Compute the difference vector
                dp = p2 - p1
                dp_norm = np.linalg.norm(dp)
                if dp_norm == 0:
                    continue
                dp /= dp_norm

                # Compute angular features
                f1 = np.dot(n1, dp)
                f2 = np.dot(n2, dp)
                f3 = np.dot(n1, n2)

                # Map features to [0, 1]
                f1 = (f1 + 1) / 2.0
                f2 = (f2 + 1) / 2.0
                f3 = (f3 + 1) / 2.0

                # Determine bin indices
                f1_bin = int(np.floor(f1 * nb_bins))
                f2_bin = int(np.floor(f2 * nb_bins))
                f3_bin = int(np.floor(f3 * nb_bins))

                # Ensure bin indices are within range
                f1_bin = min(f1_bin, nb_bins - 1)
                f2_bin = min(f2_bin, nb_bins - 1)
                f3_bin = min(f3_bin, nb_bins - 1)

                # Increment histogram bin
                hist[f1_bin, f2_bin, f3_bin] += 1

        # Normalize histogram
        hist /= np.sum(hist)
        pfh_features.append(hist.flatten())

    return np.array(pfh_features)

# Example usage
if __name__ == "__main__":
    # Load a point cloud
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/models/ricardo9/ply/frame0000.ply")
    
    # Compute PFH features
    pfh = compute_pfh(pcd, search_radius=5, nb_bins=5)
    
    # Display extracted features
    print("Extracted PFH features:")
    print(pfh.mean())

#Achar uma forma de encontrar o raio de busca ideal: search_radius=?