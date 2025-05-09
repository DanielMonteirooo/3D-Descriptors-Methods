import os
import pandas as pd
import open3d as o3d
import pymeshlab
from contextlib import redirect_stdout
import tempfile
import io
import numpy as np
from tqdm import tqdm

# Importa os métodos de descritores
#from methods.m01_spinimages import extract_spinimages
#from methods.m02_PFH import extract_pfh
#from methods.m03_FPFH import extract_fpfh
#from methods.m04_SHOT import extract_shot
from methods.C14_RGB import compute_rgb_covariance_descriptor

# =======================
# CONFIGURAÇÕES DO SCRIPT
# =======================
INPUT_DATASET_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv"
OUTPUT_FEATURES_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/saida_features.csv"
FEATURES_EXTRACTOR = "rgb_covariance"  # Altere para outros conforme necessário

# =======================
# FUNÇÕES AUXILIARES
# =======================

# Função de leitura de nuvem de pontos
def safe_read_point_cloud(path: str):
    try:
        pc = o3d.io.read_point_cloud(path)
        return pc
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None #### ATENCAO

# Mapeamento dos descritores para suas funções
def get_descriptor_function(name):
    mapping = {
        #"spinimages": extract_spinimages,
        "rgb_covariance": compute_rgb_covariance_descriptor,
        #"pfh": extract_pfh,
        #"fpfh": extract_fpfh,
        #"shot": extract_shot,
        # ...adicione outros aqui
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f'Descritor {name} não disponível.')

"""
# Função para calcular o histograma (se necessário)
def compute_histogram(feature_vector, bins=64):
    # Se o vetor já for um histograma, apenas retorne
    if len(feature_vector.shape) == 1:
        return feature_vector
    # Caso contrário, calcule o histograma global
    hist, _ = np.histogram(feature_vector, bins=bins, density=True)
    return hist
"""
def feature_extraction(pc, descriptor):
    extract_func = get_descriptor_function(descriptor)
    feature_vector = extract_func(pc)
    #feature_hist = compute_histogram(feature_vector)
    return feature_vector


def process_row(input_row, descriptor):
    ref_pc_path = os.path.join(str(input_row["REFLOCATION"]), str(input_row["REF"]))
    test_pc_path = os.path.join(str(input_row["LOCATION"]), str(input_row["SIGNAL"]))
    ref_pc = safe_read_point_cloud(ref_pc_path)
    test_pc = safe_read_point_cloud(test_pc_path)
    if ref_pc is None or test_pc is None:
        print(f"Erro ao processar linha: {input_row}")
        return None
    ref_features = feature_extraction(ref_pc, descriptor)
    test_features = feature_extraction(test_pc, descriptor)
    output_row = {
        "SIGNAL": input_row["SIGNAL"],
        "REF": input_row["REF"],
        "LOCATION": input_row["LOCATION"],
        "REFLOCATION": input_row["REFLOCATION"],
        "SCORE": input_row["SCORE"],
        "ATTACK": input_row["ATTACK"],
        "CLASS": input_row["CLASS"]
    }
    print(test_features) # teste
    for i, ref in enumerate(ref_features):
        output_row[f"fv_ref_{i}"] = ref
    for i, test in enumerate(test_features):
        output_row[f"fv_test_{i}"] = test
    return output_row

# =======================
# EXECUÇÃO PRINCIPAL
# =======================

def main():
    df_in = pd.read_csv(INPUT_DATASET_CSV)
    rows_out = []
    for _, row in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        result = process_row(row, FEATURES_EXTRACTOR)
        if result is not None:
            rows_out.append(result)
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"Arquivo gerado: {OUTPUT_FEATURES_CSV}")

if __name__ == "__main__":
    main()