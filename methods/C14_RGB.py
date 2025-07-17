import numpy as np
import open3d as o3d
import os
import glob
from contextlib import redirect_stdout
import io

def compute_rgb_covariance_descriptor(point_cloud):
    """
    Compute the RGB Covariance Descriptor for a given point cloud.
    Parameters:
    point_cloud (o3d.geometry.PointCloud): Input point cloud with color information.
    Returns:
    np.ndarray: Eigenvalues of the covariance matrix. Returns an array of zeros if no colors are present.
    
    # Garante que a nuvem de pontos tem cores. Se não, retorna zeros.
    # Isso evita que o script quebre e alinha com o comportamento do csv_script.py
    if not point_cloud.has_colors() or len(point_cloud.colors) == 0:
        print("Aviso: A nuvem de pontos não possui informações de cor. Retornando vetor de zeros.")
        # Retorna um vetor de zeros com o mesmo tamanho da saída esperada (6, neste caso).
        return np.zeros(6)
    """
    # Extract points and colors
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Combine points and colors into a single feature matrix
    features = np.hstack((points, colors))

    # Compute the mean of the features
    mean_features = np.mean(features, axis=0)

    # Center the features by subtracting the mean
    centered_features = features - mean_features

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_features, rowvar=False)

    return np.linalg.eigvals(np.nan_to_num(covariance_matrix))


# Path to your point cloud file
pvs_directory = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/PVS/"

# 2. Use glob para encontrar todos os arquivos .ply no diretório
# O asterisco (*) é um curinga que corresponde a qualquer sequência de caracteres.
ply_files = glob.glob(os.path.join(pvs_directory, "*.ply"))

# Verifica se algum arquivo foi encontrado
if not ply_files:
    print(f"Nenhum arquivo .ply foi encontrado no diretório: {pvs_directory}")
else:
    print(f"Encontrados {len(ply_files)} arquivos .ply para processar.")

# 3. Itere sobre cada arquivo encontrado e processe-o
for file_path in ply_files:
    print(f"\n--- Processando: {os.path.basename(file_path)} ---")
    
    try:
        # Tenta carregar a nuvem de pontos
        # Suprime a saída de "Read PLY failed" para manter o console limpo
        with io.StringIO() as buf, redirect_stdout(buf):
            pcd = o3d.io.read_point_cloud(file_path)
        
        # Verifica se a nuvem de pontos foi carregada corretamente
        if pcd is None or len(pcd.points) == 0:
            print("Erro: Não foi possível carregar a nuvem de pontos.")
            continue # Pula para o próximo arquivo

        # Calcula o descritor RGB Covariance
        rgb_covariance_descriptor = compute_rgb_covariance_descriptor(pcd)
        
        # Exibe o descritor
        print("Descritor RGB Covariance (Autovalores):")
        print(rgb_covariance_descriptor)

    except Exception as e:
        # Captura outras exceções que possam ocorrer durante o processamento
        print(f"Ocorreu um erro inesperado ao processar o arquivo: {e}")
