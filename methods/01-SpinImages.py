import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_spin_image(point_cloud, query_point_index, bin_size=0.01, image_width=10):
    """
    Compute the Spin-Image for a given query point in a point cloud.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        query_point_index (int): Index of the query point in the point cloud.
        bin_size (float): The size of each bin in the spin-image.
        image_width (int): The number of bins along each axis of the spin-image.

    Returns:
        np.ndarray: A 2D numpy array representing the Spin-Image.
    """
    # Ensure the point cloud has normals
    if not point_cloud.has_normals():
        raise ValueError("Point cloud must have normals computed.")

    # Extract the query point and its normal
    query_point = np.asarray(point_cloud.points)[query_point_index]
    query_normal = np.asarray(point_cloud.normals)[query_point_index]

    # Initialize the spin-image
    spin_image = np.zeros((image_width, image_width))

    # Iterate over all points in the point cloud
    for point in np.asarray(point_cloud.points):
        # Vector from query point to current point
        vector = point - query_point

        # Decompose vector into components
        alpha = np.dot(vector, query_normal)  # Projection onto normal
        beta = np.linalg.norm(np.cross(vector, query_normal))  # Perpendicular distance

        # Map alpha and beta to bin indices
        i = int(np.floor(alpha / bin_size) + image_width // 2)
        j = int(np.floor(beta / bin_size))

        # Accumulate into the spin-image if indices are within bounds
        if 0 <= i < image_width and 0 <= j < image_width:
            spin_image[j, i] += 1

    return spin_image

# Load a point cloud
pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply")

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Define the query point index
query_point_index = 100  # Example index; choose based on your data

# Compute the Spin-Image
spin_image = compute_spin_image(pcd, query_point_index, bin_size=0.01, image_width=50)

# Visualize the Spin-Image
#plt.imshow(spin_image, cmap='hot', interpolation='nearest')
#plt.title("Spin-Image")
#plt.xlabel("Alpha / Bin")
#plt.ylabel("Beta / Bin")
#plt.colorbar(label='Accumulated Count')
#plt.show()
print(spin_image.mean())

#Talvez mudar o mathplotlib para tk; pesquisar no stackoverflow