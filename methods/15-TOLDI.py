import open3d as o3d
import numpy as np

def compute_toldi_features(pcd, keypoints, radius=0.1, num_bins=10):
    """
    Computes the TOLDI (Tensor of Local Descriptors for 3D Keypoints) features.

    Parameters:
    - pcd: open3d.geometry.PointCloud, input point cloud.
    - keypoints: numpy.ndarray of shape (N, 3), keypoints where features are extracted.
    - radius: float, radius of the local neighborhood.
    - num_bins: int, number of bins for histogram computation.

    Returns:
    - features: numpy.ndarray of shape (N, num_bins * 3), TOLDI descriptors for keypoints.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)

    features = []

    for kp in keypoints:
        [_, idx, _] = kdtree.search_radius_vector_3d(kp, radius)
        neighbors = points[idx]

        if len(neighbors) < 5:
            # Avoid poorly populated neighborhoods
            features.append(np.zeros(num_bins * 3))
            continue

        # Center neighbors around the keypoint
        centered_neighbors = neighbors - kp

        # Compute histograms for each axis
        histograms = []
        for axis in range(3):
            values = centered_neighbors[:, axis]
            hist, _ = np.histogram(values, bins=num_bins, range=(-radius, radius))
            histograms.append(hist)

        # Flatten histograms into a single descriptor
        descriptor = np.concatenate(histograms)
        features.append(descriptor)

    return np.array(features)

# Example Usage
# Load a sample point cloud
pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")

# Detect keypoints using ISS (Intrinsic Shape Signatures) method
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

# Convert keypoints to numpy array
keypoint_coords = np.asarray(keypoints.points)

# Compute TOLDI features
toldi_features = compute_toldi_features(pcd, keypoint_coords)

# Display results
for i, feature in enumerate(toldi_features):
    print(f"Keypoint {i}: Feature Vector {feature}")
