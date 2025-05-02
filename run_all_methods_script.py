import subprocess
import os
import time

def executar_script(caminho_completo, arquivo):
    try:
        inicio = time.time()
        subprocess.run(
            ["python3", caminho_completo],
            capture_output=True,
            text=True,
            check=True,
            timeout=100
        )
        fim = time.time()
        tempo_execucao = fim - inicio
        print(f"Método '{arquivo}' executado com sucesso.")
        print(f"Tempo de execução: {tempo_execucao:.3f} segundos")
        print("-" * 50)
        return tempo_execucao
    except subprocess.TimeoutExpired:
        print(f"Tempo limite excedido ao executar '{arquivo}'")
        print("-" * 50)
        return "Erro: Tempo limite excedido"
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar '{arquivo}': {e}")
        print("-" * 50)
        return f"Erro: {e}"
    except Exception as e:
        print(f"Erro inesperado ao executar '{arquivo}': {e}")
        print("-" * 50)
        return f"Erro: {e}"

def listar_scripts(pasta):
    return sorted([
        arquivo for arquivo in os.listdir(pasta)
        if arquivo.endswith(".py")
    ])

def exibir_opcoes_em_duas_colunas(opcoes):
    largura_coluna = max(len(opcao) for opcao in opcoes) + 4  # Ajusta largura para alinhamento
    colunas = 2  # Número de colunas desejadas
    linhas = (len(opcoes) + colunas - 1) // colunas  # Calcula número de linhas

    for i in range(linhas):
        linha = ""
        for j in range(colunas):
            indice = i + j * linhas
            if indice < len(opcoes):
                linha += f"{indice + 1} - {opcoes[indice]:<{largura_coluna}}"
        print(linha)

def menu_interativo(pasta):
    arquivos_python = listar_scripts(pasta)
    
    if not arquivos_python:
        print("Nenhum arquivo .py encontrado na pasta.")
        return
    
    print("Selecione uma opção:")
    print("0 - Executar todos os scripts")
    print("=============================")
    exibir_opcoes_em_duas_colunas(arquivos_python)
    
    escolha = input("Digite o número correspondente à sua escolha: ")
    
    try:
        escolha = int(escolha)
        
        if escolha == 0:
            resultados = {}
            for arquivo in arquivos_python:
                caminho_completo = os.path.join(pasta, arquivo)
                resultado = executar_script(caminho_completo, arquivo)
                resultados[arquivo] = resultado
            
            print("\nResumo dos Tempos de Execução:")
            print("=" * 50)
            for arquivo, tempo in resultados.items():
                if isinstance(tempo, float):
                    print(f"- {arquivo}: {tempo:.3f} segundos")
                else:
                    print(f"- {arquivo}: {tempo}")
        
        elif 1 <= escolha <= len(arquivos_python):
            arquivo_escolhido = arquivos_python[escolha - 1]
            caminho_completo = os.path.join(pasta, arquivo_escolhido)
            executar_script(caminho_completo, arquivo_escolhido)
        
        else:
            print("Opção inválida. Tente novamente.")
    
    except ValueError:
        print("Entrada inválida. Por favor, insira um número.")

# Caminho para a pasta dos scripts
pasta_dos_scripts = "/home/dani/Estudos/PIBIC/3D-Descriptors/methods/"
menu_interativo(pasta_dos_scripts)
