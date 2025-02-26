import open3d as o3d
import numpy as np

def compute_color_ppf(point_cloud):
    """
    Compute Color Point Pair Features (Color-PPF) for a given point cloud.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud with normals and colors.
        
    Returns:
        features (list): List of Color-PPF features.
    """
    if not point_cloud.has_normals():
        raise ValueError("Point cloud must have normals. Use estimate_normals() first.")
    if not point_cloud.has_colors():
        raise ValueError("Point cloud must have colors.")

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    colors = np.asarray(point_cloud.colors)
    
    features = []
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            n1, n2 = normals[i], normals[j]
            c1, c2 = colors[i], colors[j]
            
            # Compute geometric features
            d = p2 - p1
            d_norm = np.linalg.norm(d)
            if d_norm == 0:  # Skip if the distance is zero
                continue
            d_unit = d / d_norm
            
            angle_n1_d = np.arccos(np.clip(np.dot(n1, d_unit), -1.0, 1.0))
            angle_n2_d = np.arccos(np.clip(np.dot(n2, d_unit), -1.0, 1.0))
            angle_n1_n2 = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
            
            # Combine geometric and color information
            color_diff = np.linalg.norm(c1 - c2)
            feature = (d_norm, angle_n1_d, angle_n2_d, angle_n1_n2, color_diff)
            features.append(feature)
    
    return features

# Main execution
if __name__ == "__main__":
    # Path to the PLY file
    file_path = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_romanoillamp_vox10_dec_geom02_text02_trisoup-predlift.ply"

    # Load the point cloud from the specified path
    point_cloud = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud was loaded successfully
    if not point_cloud.is_empty():
        print("Point cloud loaded successfully.")
        
        # Estimate normals for the point cloud
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Compute Color-PPF features
        color_ppf_features = compute_color_ppf(point_cloud)

        # Display the extracted features
        print(f"Extracted {len(color_ppf_features)} Color-PPF features.")
        for i, feature in enumerate(color_ppf_features[:10]):  # Display first 10 features
            print(f"Feature {i + 1}: {feature}")
    else:
        print("Failed to load the point cloud. Please check the file path.")

#Error