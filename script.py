import subprocess
import os
import time

def executar_script(caminho_completo, arquivo):
    try:
        inicio = time.time()
        
        # Executa o script e captura a saída
        subprocess.run(
            ["python3", caminho_completo],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        fim = time.time()
        tempo_execucao = fim - inicio
        
        print(f"Script '{arquivo}' executado com sucesso.")
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

def executar_scripts(pasta):
    resultados = {}
    arquivos_python = sorted([
        arquivo for arquivo in os.listdir(pasta)
        if arquivo.endswith(".py")
    ])

    print("Iniciando execução dos scripts:")
    print("=" * 50)

    for arquivo in arquivos_python:
        caminho_completo = os.path.join(pasta, arquivo)
        resultado = executar_script(caminho_completo, arquivo)
        resultados[arquivo] = resultado

    return resultados

pasta_dos_scripts = "/home/dani/Estudos/PIBIC/3D-Descriptors/methods/"
resultados_obtidos = executar_scripts(pasta_dos_scripts)

print("\nResumo dos Tempos de Execução:")
print("=" * 50)
for arquivo, tempo in resultados_obtidos.items():
    if isinstance(tempo, float):
        print(f"- {arquivo}: {tempo:.3f} segundos")
    else:
        print(f"- {arquivo}: {tempo}")
