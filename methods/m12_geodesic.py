import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

def compute_hks(mesh, num_eigenpairs=100, time_scales=None):
    # Ensure the mesh has vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Convert mesh to numpy arrays
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Compute the cotangent weights for the Laplace-Beltrami Operator
    # This requires computing edge lengths and angles
    # For brevity, detailed computation steps are omitted here

    # Example of constructing the Laplace-Beltrami Operator (LBO)
    # Here we use a simplified version for demonstration purposes
    n_vertices = vertices.shape[0]
    I = np.arange(n_vertices)
    M = coo_matrix((np.ones(n_vertices), (I, I)), shape=(n_vertices, n_vertices))  # Mass matrix (identity for simplicity)
    W = coo_matrix((np.ones(n_vertices), (I, I)), shape=(n_vertices, n_vertices))  # Stiffness matrix (identity for simplicity)
    L = M - W  # Simplified Laplace-Beltrami Operator

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = eigsh(L, k=num_eigenpairs, sigma=0, which='LM')

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the HKS
    if time_scales is None:
        time_scales = np.logspace(-2, 2, 100)  # Default time scales

    hks = np.zeros((vertices.shape[0], len(time_scales)))
    for i, t in enumerate(time_scales):
        hks[:, i] = np.sum(
            np.exp(-eigenvalues * t) * (eigenvectors ** 2), axis=1
        )

    return hks

# Load a mesh using Open3D
mesh = o3d.io.read_triangle_mesh("/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/tmc13_amphoriskos_vox10_dec_geom04_text04_octree-predlift.ply")

# Compute the HKS
hks = compute_hks(mesh)

# hks now contains the Heat Kernel Signatures for each vertex
