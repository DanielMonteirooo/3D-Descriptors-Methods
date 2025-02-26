import pandas as pd

# Carrega o arquivo CSV em um DataFrame
df = pd.read_csv('/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv')

# Gera uma lista de dicionários a partir do DataFrame
lista_dicionarios = df.to_dict(orient='records')

# Itera sobre a lista de dicionários
for index, linha in enumerate(lista_dicionarios):
    print(f"Linha {index}:")
    print(f"  SIGNAL: {linha['SIGNAL']}")
    print(f"  LOCATION: {linha['LOCATION']}")
    print(f"  REFLOCATION: {linha['REFLOCATION']}")
    print("---")

#referencia:
#rof_path = os.path.join(df[df["ref_location],df["ref"]])
#test_path = os.path.join(df[df["location"],df["signal"]])
#ref = o3d.io.read_point_cloud(rof_path)
#test = o3d.io.read_point_cloud(test_path)
#ref_feat= [stot(ref)]


'''
# Para realizar operações específicas 

import os

for linha in lista_dicionarios:
    caminho_pvs = linha['LOCATION']
    caminho_ref = linha['REFLOCATION']
    
    # Exemplo: verificar se os arquivos existem
    if os.path.exists(caminho_pvs) and os.path.exists(caminho_ref):
        print(f"Arquivos encontrados para {linha['SIGNAL']}")
    else:
        print(f"Arquivos não encontrados para {linha['SIGNAL']}")

'''