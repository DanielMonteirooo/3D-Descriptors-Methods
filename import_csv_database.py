import pandas as pd

# Caminhos para os arquivos CSV
input_csv_path = '/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa.csv'
output_csv_path = '/home/dani/Estudos/PIBIC/APSIPA___M-PCCD/apsipa_processed.csv'

# Carrega o arquivo CSV de entrada em um DataFrame
df = pd.read_csv(input_csv_path)

# Processa cada linha do DataFrame e reorganiza os dados conforme especificado
processed_rows = []

for _, input_row in df.iterrows():
    # Cria um dicionário com a ordem especificada
    output_row = {
        "REF": input_row["REF"],
        "SIGNAL": input_row["SIGNAL"],
        "SCORE": input_row["SCORE"],
        "ATTACK": input_row["ATTACK"],
        "CLASS": input_row["CLASS"]
    }
    
    # Adiciona a linha processada à lista
    processed_rows.append(output_row)

# Cria um novo DataFrame com as linhas processadas
df_processed = pd.DataFrame(processed_rows)

# Salva o DataFrame processado em um novo arquivo CSV
df_processed.to_csv(output_csv_path, index=False)

print(f"Arquivo CSV processado gerado com sucesso em: {output_csv_path}")
