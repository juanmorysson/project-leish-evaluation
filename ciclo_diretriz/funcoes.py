import os, re
import pandas as pd
import chardet
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

tempo_delay = 30

def obter_dados_paciente(numero_paciente, path):
  path_resultados = path+"resultados_shap.csv"
  path_dataframe = path+"dataset_processado_437.csv"
  path_dicionario = path+"dicionario.csv"

  with open(path_dicionario, 'rb') as f:
    result = chardet.detect(f.read())
    print(f"Encoding detectado: {result['encoding']} com confiança {result['confidence']}")

  # Leitura dos arquivos
  df_resultados = pd.read_csv(path_resultados)
  df_imputado = pd.read_csv(path_dataframe, sep=';')
  dicionario_df = pd.read_csv(path_dicionario, delimiter=';', encoding=result['encoding'])

  # Verificar duplicatas
  duplicatas = dicionario_df[dicionario_df['Variável'].duplicated(keep=False)]

  #print("\nÚltimas 10 linhas do dicionário:")
  #print(dicionario_df.tail(10))

  if len(duplicatas) > 0:
      #print(f"\nEncontradas {len(duplicatas)} linhs duplicadas:")
      #print("=" * 80)

      # Agrupar por variável para mostrar todas as ocorrências
      variaveis_duplicadas = duplicatas['Variável'].unique()

      #for var in variaveis_duplicadas:
          #ocorrencias = dicionario_df[dicionario_df['Variável'] == var]
          #print(f"\nVariável: '{var}' - {len(ocorrencias)} ocorrências:")
          #for idx, row in ocorrencias.iterrows():
              #print(f"   Linha {idx}:")
              #print(f"   - Descrição: {row.get('Descrição', 'N/A')}")
              #print(f"   - Importância: {row.get('Importância esperada', 'N/A')}")
              #print(f"   - Observação: {row.get('Observação', 'N/A')}")
              #print(f"   - Tipo Estatística: {row.get('Tipo Estatística', 'N/A')}")

      #print("\n" + "=" * 80)
      #print("Removendo duplicatas (mantendo primeira ocorrência)...")

      # Manter apenas a primeira ocorrência de cada variável duplicada
      dicionario_df = dicionario_df.drop_duplicates(subset=['Variável'], keep='first')
      #print(f"Dicionário após remover duplicatas: {len(dicionario_df)} linhas")


  # Criar dicionário para busca rápida
  dicionario_dict = dicionario_df.set_index('Variável').to_dict('index')

  # Exibir os nomes das colunas principais para verificação
  #print("Colunas do resultados:", df_resultados.columns[:10].tolist())
  #print("Colunas do imputado:", df_imputado.columns[:10].tolist())

  # Converter colunas numéricas para float, ignorando as que não são numéricas
  colunas_excluir = ['index', 'base_value', 'predicao', 'waterfall_image']
  for col in df_resultados.columns:
      if col not in colunas_excluir:
          df_resultados[col] = pd.to_numeric(df_resultados[col], errors='coerce')


  #numero_paciente = int(input("\n\nDigite o número do paciente (índice da linha): "))
  numero_paciente = int(numero_paciente)
  linha_resultado = df_resultados.iloc[numero_paciente]
  linha_imputado = df_imputado.iloc[numero_paciente]

  # Converter a linha imputada para dicionário
  linha_dict = linha_imputado.to_dict()

  # Selecionar apenas as colunas SHAP numéricas
  colunas_shap = [c for c in df_resultados.columns if c not in colunas_excluir]
  valores_shap = linha_resultado[colunas_shap].astype(float)

  # Pegar os 9 (ou 5) maiores valores absolutos de SHAP
  top_n = 4
  top_features = valores_shap.abs().nlargest(top_n).index.tolist()

  def buscar_info_dicionario(variavel):
      if variavel in dicionario_dict:
          info = dicionario_dict[variavel]

          observacao = info.get('Observação', '')
          if pd.isna(observacao) or observacao == '':
              observacao = 'Nenhuma'

          return {
              'descricao': info.get('Descrição', 'Descrição não encontrada'),
              'importancia': info.get('Importância esperada', 'Importância não definida'),
              'observacao': observacao
          }
      else:
          return {
              'descricao': 'Descrição não encontrada',
              'importancia': 'Importância não definida',
              'observacao': 'Nenhuma'
          }

  # Criar string formatada automaticamente com valores SHAP, reais E informações do dicionário
  shap_info = []
  for f in top_features:
      shap_val = linha_resultado[f]
      val_real = linha_dict.get(f, 'N/A')

      # Buscar informações no dicionário
      info_dict = buscar_info_dicionario(f)

      # Formatar a linha com todas as informações
      linha = f"{f} = {val_real} (impacto SHAP = {shap_val:.5f})"
      linha += f" | Descrição: {info_dict['descricao']}"
      linha += f" | Importância: {info_dict['importancia']}"
      if info_dict['observacao'] and info_dict['observacao'] != 'Nenhuma':
          linha += f" | Observação: {info_dict['observacao']}"

      shap_info.append(linha)

  SHAPE_VALUES = "\n".join(shap_info)

  # Obter risco (predição final) e valor base
  risco = linha_resultado.get('predicao', linha_imputado.get('probMorte', 'desconhecido'))
  base_value = linha_resultado.get('base_value', 'desconhecido')
  return SHAPE_VALUES, numero_paciente, risco
    
# buscando diretriz atual
def busca_diretriz(rodada, path):
  csv_file = 'diretrizes.csv'
  df_diterizes = pd.read_csv(path+csv_file, sep=';', encoding='latin-1')
  dicionario_dict = df_diterizes.set_index('rodada').to_dict('index')
  info = dicionario_dict[int(rodada)]
  return info.get('diretriz')
  
def add_resposta(path, rodada, anotador, paciente, resposta):
    csv_file = path+'respostas.csv'
    df_respostas = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    nova_linha_dict = {
        'rodada': rodada,
        'anotador': anotador,
        'paciente': paciente,
        'resposta': resposta,
        'sugestao': ''
    }
    nova_linha = pd.DataFrame([nova_linha_dict])
    df_atualizado = pd.concat([df_respostas, nova_linha], ignore_index=True)
    df_atualizado.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
    print("Resposta de {anotador} adicionada com sucesso!")
    time.sleep(tempo_delay)
    print("passou tempo")
    
def add_critica(path, rodada, anotador, paciente, critica):
  csv_file = path+'respostas.csv'
  df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
  mascara = (df['rodada'] == int(rodada)) & (df['anotador'] == anotador) & (df['paciente'] == int(paciente))
  if mascara.any():
    df['critica'] = df['critica'].astype(object)
    df.loc[mascara, 'critica'] = critica
    #salvar_csv_com_garantia(df=df,caminho=csv_file)
    df.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
    print("Critica adicionada!")
    time.sleep(tempo_delay)
    print("passou tempo")
  else:
    print("Não achou linha da anotação pra alterar c")

def add_sugestao(path, rodada, anotador, paciente, novo_valor_sugestao):
  csv_file = path+'respostas.csv'
  df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
  mascara = (df['rodada'] == rodada) & (df['anotador'] == anotador) & (df['paciente'] == paciente)
  if mascara.any():
    df['sugestao'] = df['sugestao'].astype(object)
    df.loc[mascara, 'sugestao'] = novo_valor_sugestao
    #salvar_csv_com_garantia(df=df,caminho=csv_file)
    df.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
    print("Sugestão adicionada!")
    time.sleep(tempo_delay)
    print("passou tempo")
  else:
    print("Não achou linha da anotação pra alterar")

def busca_sugestoes(rodada, path):
    csv_file = path + 'respostas.csv'
    df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    series_sugestao = df[df['rodada'] == int(rodada)]['sugestao']
    series_sugestao_limpa = series_sugestao.dropna()
    lista_com_vazias = series_sugestao_limpa.astype(str).tolist()
    lista_final_filtrada = [sugestao for sugestao in lista_com_vazias if sugestao.strip()]
    return lista_final_filtrada
    
def add_diretriz(path, rodada, diretriz):
    csv_file = path+'diretrizes.csv'
    df_diretrizes = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    nova_linha_dict = {
        'rodada': int(rodada)+1,
        'diretriz': diretriz
    }
    nova_linha = pd.DataFrame([nova_linha_dict])
    df_atualizado = pd.concat([df_diretrizes, nova_linha], ignore_index=True)
    #salvar_csv_com_garantia(df=df_atualizado,caminho=csv_file)
    df_atualizado.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
    print("Nova Diretriz adicionada com sucesso!")
    time.sleep(tempo_delay)
    print("passou tempo")

def medir_similaridade(paciente, path):
    csv_file = path + 'respostas.csv'
    df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    respostas = df[df['paciente'] == int(paciente)][['anotador','resposta']]
    respostas_limpa = respostas.dropna()
    print("Paciente:" +str(paciente))
    print("Qtd respostas:" +str(len(respostas_limpa)))
    #print(respostas_limpa)
    #print(df.head(10))
    series_respostas = respostas_limpa['resposta']
    lista_respostas = series_respostas.astype(str).tolist()
    similaridades = []
    if len(respostas_limpa) > 0:
        # 'all-MiniLM-L6-v2' é um bom modelo para similaridade semântica
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(lista_respostas)
        similarity_matrix = cosine_similarity(embeddings)
        pares_indices = list(itertools.combinations(respostas_limpa.index, 2))
        tamanho_passo = len(respostas_limpa)
        quant = 0
        x, y = pares_indices[0]
        while x >= tamanho_passo:
            x = x - tamanho_passo
            quant = quant + 1
        for i, j in pares_indices:
            anotador1 = df.loc[i, 'anotador']
            anotador2 = df.loc[j, 'anotador']
            similaridade_score = similarity_matrix[i - tamanho_passo*quant, j - tamanho_passo*quant]
            similaridades.append(similaridade_score)
            print('Similaridade Coseno das respostas de '+anotador1+' e '+anotador2+': '+str(similaridade_score))
    media = np.mean(similaridades)
    print(f"A Média das similaridades do paciente "+str(paciente)+f" é: {media:.4f}")
    return media

def ler_variaveis_do_arquivo(path):
    nome_arquivo = path + 'rodada.txt'
    valores_encontrados = {}
    padrao = re.compile(r'^\s*#?\s*(\w+)\s*=\s*(\S+)')

    try:
        with open(nome_arquivo, 'r') as f:
            for linha in f:
                linha = linha.strip()
                match = padrao.match(linha)
                if match:
                    nome_variavel = match.group(1)
                    valor = match.group(2)
                    if nome_variavel in ['rodada', 'paciente']:
                        if nome_variavel not in valores_encontrados or not linha.startswith('#'):
                            try:
                                valores_encontrados[nome_variavel] = int(valor)
                            except ValueError:
                                valores_encontrados[nome_variavel] = valor

        return valores_encontrados.get('rodada', 'Não encontrado'), valores_encontrados.get('paciente', 'Não encontrado')

    except FileNotFoundError:
        return {"erro": f"O arquivo '{nome_arquivo}' não foi encontrado."}
    
def add_resposta_final(path, anotador, paciente, resposta):
    csv_file = path+'respostas_'+anotador+'.csv'
    df_respostas = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    nova_linha_dict = {
        'paciente': paciente,
        'resposta': resposta,
    }
    nova_linha = pd.DataFrame([nova_linha_dict])
    df_atualizado = pd.concat([df_respostas, nova_linha], ignore_index=True)
    #salvar_csv_com_garantia(df=df_atualizado,caminho=csv_file)
    df_atualizado.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
    print("Paciente "+str(paciente)+" ok!")
    time.sleep(tempo_delay)
    print("passou tempo")

def medir_similaridade_final(paciente, path):
    csv_file = path + 'respostas_final.csv'
    df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    linha = df[df['paciente'] == int(paciente)]
    lista_respostas = []
    print(type(linha['Amanda'].item()))
    print(type(lista_respostas))
    lista_respostas.append(linha['Amanda'].item())
    lista_respostas.append(linha['Beatriz'].item())
    lista_respostas.append(linha['Carlos'].item())
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(lista_respostas)
    similarity_matrix = cosine_similarity(embeddings)
    axb = similarity_matrix[0, 1]
    axc = similarity_matrix[0, 2]
    bxc = similarity_matrix[1, 2]
    media = np.mean([axb, axc, bxc])
    mascara = (df['paciente'] == int(paciente))
    if mascara.any():
        df.loc[mascara, 'axb'] = axb
        df.loc[mascara, 'axc'] = axc
        df.loc[mascara, 'bxc'] = bxc
        df.loc[mascara, 'media'] = media
        #salvar_csv_com_garantia(df=df,caminho=csv_file)
        df.to_csv(csv_file, sep=';', index=False, encoding='latin-1', errors='ignore')
        print(f"Paciente "+str(paciente)+" similaridade ok")
        time.sleep(tempo_delay)
        print("passou tempo")
    else:
        print("Não achou linha para alterar")

def medir_similaridade_testes(paciente, path, llm):
    csv_file_llm = path + 'respostas_'+llm+'.csv'
    df_llm = pd.read_csv(csv_file_llm, sep=';', encoding='latin-1')
    linha_llm = df_llm[df_llm['paciente'] == int(paciente)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    csv_file = path + 'respostas_final.csv'
    df = pd.read_csv(csv_file, sep=';', encoding='latin-1')
    linha = df[df['paciente'] == int(paciente)]
    lista_respostas = []
    lista_respostas.append(linha['Amanda'].item())
    lista_respostas.append(linha_llm['resposta'].item())
    embeddings = model.encode(lista_respostas)
    similarity_matrix = cosine_similarity(embeddings)
    sim = similarity_matrix[0, 1]
    print("sim_a: ")
    print(sim)
    
    lista_respostas = []
    lista_respostas.append(linha['Beatriz'].item())
    lista_respostas.append(linha_llm['resposta'].item())
    embeddings = model.encode(lista_respostas)
    similarity_matrix = cosine_similarity(embeddings)
    sim = similarity_matrix[0, 1]
    print("sim_b: ")
    print(sim)

    lista_respostas = []
    lista_respostas.append(linha['Carlos'].item())
    lista_respostas.append(linha_llm['resposta'].item())
    embeddings = model.encode(lista_respostas)
    similarity_matrix = cosine_similarity(embeddings)
    sim = similarity_matrix[0, 1]
    print("sim_c: ")
    print(sim)