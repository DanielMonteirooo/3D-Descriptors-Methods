import os
import pandas as pd
import open3d as o3d
from contextlib import redirect_stdout
import tempfile
import io
import numpy as np
import pymeshlab
from tqdm import tqdm

# Importa os métodos de descritores
# from methods.m01_spinimages import extract_spinimages
# from methods.m02_PFH import extract_pfh
# from methods.m03_FPFH import extract_fpfh
# from methods.m04_SHOT import extract_shot
from methods.C14_RGB import compute_rgb_covariance_descriptor

# =======================
# CONFIGURAÇÕES DO SCRIPT
# =======================
INPUT_DATASET_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv"
OUTPUT_FEATURES_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/saida_features.csv"
FEATURES_EXTRACTOR = "rgb_covariance" # Altere para outros conforme necessário


# =======================
# FUNÇÕES AUXILIARES
# =======================

def safe_read_point_cloud(path: str):
    
    
    """
    Lê nuvem de pontos de um arquivo .ply usando Open3D. 
    Se falhar, converte o arquivo para um formato compatível via pymeshlab e tenta novamente.
    """
    def get_suitable_version_path(ply_path, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_filepath = os.path.join(temp_dir, "temp.ply")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(ply_path)
        ms.save_current_mesh(temp_ply_filepath)
        return temp_ply_filepath

    with tempfile.TemporaryDirectory() as temp_dir:
        with io.StringIO() as buf, redirect_stdout(buf):
            pc = o3d.io.read_point_cloud(path)
            output = buf.getvalue()
        if "Read PLY failed" in output:
            try:
                temp_ply = get_suitable_version_path(path, temp_dir)
                pc = o3d.io.read_point_cloud(temp_ply)
            except Exception as e:
                print(f"Erro ao tentar converter {path}: {e}")
                return None
        
        print("==========================", len (np.array(pc.points)))

        return pc

def get_descriptor_function(name):
    mapping = {
        # "spinimages": extract_spinimages,
        "rgb_covariance": compute_rgb_covariance_descriptor,
        # "pfh": extract_pfh,
        # "fpfh": extract_fpfh,
        # "shot": extract_shot,
        # ...adicione outros aqui
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f'Descritor {name} não disponível.')

def feature_extraction(pc, descriptor):
    extract_func = get_descriptor_function(descriptor)
    feature_vector = extract_func(pc)
    # feature_hist = compute_histogram(feature_vector)
    return feature_vector

def process_row(input_row, descriptor): 
    #Colunas originais do apsipa.CSV:
    #SIGNAL,REF,SCORE,LOCATION,REFLOCATION,ATTACK,CLASS
    #Colunas do CSV do UnB_PC_IMGS.csv:
    #SIGNAL,SCORE,SCORE_STD,LOCATION,REF,ATTACK,CLASS
    ref_pc_path = os.path.join(str(input_row["REFLOCATION"]), str(input_row["REF"]))
    test_pc_path = os.path.join(str(input_row["LOCATION"]), str(input_row["SIGNAL"]))

    ref_pc = safe_read_point_cloud(ref_pc_path)
    test_pc = safe_read_point_cloud(test_pc_path)
    if ref_pc is None or test_pc is None:
        print(f"Erro ao processar linha: {input_row}")
        print(f"ref={ref_pc}", ref_pc_path)
        print(f"test={test_pc}", test_pc_path)
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
    # Para cada feature extraída, adiciona no output_row
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

    # Inverta a ordem das linhas:
    df_in = df_in.iloc[::-1].reset_index(drop=True)
    
    for _, row in tqdm(df_in.iterrows(), total=df_in.shape[0]):
        result = process_row(row, FEATURES_EXTRACTOR)
        if result is not None:
            rows_out.append(result)
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"Arquivo gerado: {OUTPUT_FEATURES_CSV}")

if __name__ == "__main__":
    main()
