import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_spin_image(point_cloud, query_point_index, bin_size=0.01, image_width=10):
    """
    Compute the Spin-Image descriptor for a 3D point using Open3D.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud
        query_point_index (int): Index of query point
        bin_size (float): Spatial resolution of the descriptor
        image_width (int): Size of the square spin-image matrix
    
    Returns:
        np.ndarray: 2D spin-image matrix
    """
    if not point_cloud.has_normals():
        raise ValueError("Point cloud must have computed normals")
        
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    
    query_point = points[query_point_index]
    query_normal = normals[query_point_index]
    
    spin_image = np.zeros((image_width, image_width))
    
    for point in points:
        vector = point - query_point
        alpha = np.dot(vector, query_normal)
        beta = np.linalg.norm(np.cross(vector, query_normal))
        
        i = int(alpha/bin_size + image_width//2)
        j = int(beta/bin_size)
        
        if 0 <= i < image_width and 0 <= j < image_width:
            spin_image[j, i] += 1
            
    return spin_image

def visualize_spin_image(spin_image):
    """Visualize the spin-image using matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(spin_image, cmap='viridis', interpolation='nearest')
    plt.title('Spin-Image Visualization')
    plt.xlabel('Alpha (Normal projection)')
    plt.ylabel('Beta (Tangent distance)')
    plt.colorbar(label='Point Density')
    plt.show()

def calculate_averages(spin_image):
    """Calculate row/column averages and their combined average"""
    row_avg = np.mean(spin_image, axis=1)
    col_avg = np.mean(spin_image, axis=0)
    combined_avg = (np.mean(row_avg) + np.mean(col_avg)) / 2
    return row_avg, col_avg, combined_avg

if __name__ == "__main__":
    # Load and prepare point cloud
    pcd = o3d.io.read_point_cloud("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_the20smaria_00600_vox10_dec_geom06_text06_trisoup-predlift.ply")  # Replace with your file
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    
    # Compute spin-image
    spin_image = compute_spin_image(pcd, 100, bin_size=0.01, image_width=50)
    
    # Visualization
    #visualize_spin_image(spin_image)
    
    # Calculate averages
    row_avg, col_avg, final_avg = calculate_averages(spin_image)
    print(f"Row averages mean: {np.mean(row_avg):.2f}")
    print(f"Column averages mean: {np.mean(col_avg):.2f}")
    print(f"Combined average: {final_avg:.2f}")
