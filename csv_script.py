import os
import pandas as pd
import open3d as o3d
import pymeshlab
from contextlib import redirect_stdout
import tempfile
import io
import numpy as np
from tqdm import tqdm

# Importa os descritores do pacote lbplib3d
from lbplib3d.olbp import olbp_feature
from lbplib3d.ulbp import ulbp_feature
from lbplib3d.nriulbp import nriulbp_feature
from lbplib3d.rorlbp import rorlbp_feature
from lbplib3d.ltp import ltp_feature
from lbplib3d.clbp import clbp_feature
from lbplib3d.cltp import cltp_feature
from lbplib3d.cslbp import cslbp_feature

# =======================
# CONFIGURAÇÕES DO SCRIPT
# =======================
INPUT_DATASET_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv"
OUTPUT_FEATURES_CSV = "/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/saida_features.csv"
FEATURES_EXTRACTOR = "olbp"  # Altere para outros conforme necessário

# =======================
# FUNÇÕES AUXILIARES
# =======================

def feature_extraction(pc, descriptor):
    fun = function_name_to_callable(descriptor)
    # Checa se a point cloud tem cor
    if descriptor in ["olbp", "ulbp", "cslbp", "nriulbp", "rorlbp", "clbp", "cltp"]:
        if not pc.has_colors():
            raise ValueError(f"O descritor {descriptor} exige cor na nuvem de pontos.")
    return fun(pc)


def safe_read_point_cloud(path: str):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with io.StringIO() as buf, redirect_stdout(buf):
                pc = o3d.io.read_point_cloud(path)
                output = buf.getvalue()
                if "Read PLY failed" in output:
                    temp_ply = os.path.join(temp_dir, "temp.ply")
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(path)
                    ms.save_current_mesh(temp_ply)
                    pc = o3d.io.read_point_cloud(temp_ply)
        return pc
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return None

def function_name_to_callable(function_name: str):
    mapping = {
        "olbp": olbp_feature,
        "cslbp": cslbp_feature,
        "ulbp": ulbp_feature,
        "nriulbp": nriulbp_feature,
        "rorlbp": rorlbp_feature,
        "rltp": ltp_feature,
        "clbp": clbp_feature,
        "cltp": cltp_feature
    }
    if function_name in mapping:
        return mapping[function_name]
    raise ValueError(f'{function_name} não é um descritor disponível.')

def feature_extraction(pc, descriptor):
    fun = function_name_to_callable(descriptor)
    return fun(pc)

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
