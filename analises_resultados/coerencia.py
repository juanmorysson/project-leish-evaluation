from bert_score import BERTScorer
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import sys
import os
import logging
import transformers
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ciclo_diretriz import funcoes as f
os.environ["HF_TOKEN"] = ""

path = "../ciclo_diretriz/"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
transformers.utils.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# Silencia logs de detecção de encoding (muito comum em bibliotecas como chardet ou charset_normalizer)
logging.getLogger('charset_normalizer').setLevel(logging.ERROR)
logging.getLogger('chardet').setLevel(logging.ERROR)

#model_name = 'neuralmind/bert-base-portuguese-cased'
model_name = 'neuralmind/bert-base-portuguese-cased'
# 1. Carregamos a config e corrigimos o valor astronômico manualmente
config = AutoConfig.from_pretrained(model_name)
config.max_position_embeddings = 512

# 2. Carregamos o modelo e tokenizer com a config corrigida
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Forçamos o tokenizer a entender o limite
tokenizer.model_max_length = 512 

model = AutoModel.from_pretrained(model_name, config=config)

# 3. Criamos o Scorer. 
# IMPORTANTE: Passamos lang="pt", mas evitamos que ele carregue o modelo sozinho
scorer = BERTScorer(
    lang="pt", 
    model_type=model_name, 
    num_layers=9,
    device='cpu'
    )

# 4. HACK: Substituímos o modelo interno do Scorer pelo nosso BERTimbau corrigido
scorer._model = model
scorer._tokenizer = tokenizer

def verificar_coerencia_atômica(resposta_candidata, valor_risco, valores_shap, threshold=0.75):
    # Lista de fatos objetivos
    fatos = [f"O risco de mortalidade é de {valor_risco:.2f}%."]
    linhas_shap = [linha.strip() for linha in valores_shap.split('\n') if linha.strip()]
    for linha in linhas_shap:
        parte_tecnica = linha.split('|')[0].strip()
        fatos.append(parte_tecnica)

    scores_fatos = []
    with torch.no_grad():
        for fato in fatos:
            print(fato)
            # Comparamos cada fato individualmente contra a resposta
            P, R, F1 = scorer.score([resposta_candidata], [fato])
            scores_fatos.append(R.item())
    
    # A média ou o valor mínimo dirá se os dados técnicos foram preservados
    nota_final = np.mean(scores_fatos)
    return nota_final > threshold, nota_final

def verificar_coerencia_bertscore(resposta_candidata, valor_risco, valores_shap, threshold=0.6):
    """
    Verifica se a resposta do LLM é coerente com os dados brutos do SHAP e Risco.
    
    Args:
        resposta_candidata (str): O texto gerado pelo modelo.
        valor_risco (float): A probabilidade de morte (ex: 15.5).
        valores_shap (dict): Dicionário {feature_name: shap_value}.
        threshold (float): Nota mínima de F1-Score para considerar a resposta coerente.
        
    Returns:
        tuple: (bool: coerente, float: f1_score, str: ancora)
    """
    texto_shap = ""
    linhas_shap = [linha.strip() for linha in valores_shap.split('\n') if linha.strip()]
    for linha in linhas_shap:
        parte_tecnica = linha.split('|')[0].strip()
        texto_shap = texto_shap + parte_tecnica
    txt_ancora = f"Risco: {valor_risco:.2f}%. SHAP Waterfall: {texto_shap}."
    
    # Removendo espaços extras e garantindo string
    candidato = str(resposta_candidata).strip()
    referencia = str(txt_ancora).strip()

    # Cálculo
    with torch.no_grad():
        P, R, F1 = scorer.score([candidato], [referencia], verbose=False)
    
    nota = R.item()
    coerente = nota > threshold
    
    return coerente, nota

df_input = pd.read_csv(path+"respostas_melhor_diretriz_corrigido.csv") # Deve conter paciente, resposta_a, resposta_b, resposta_c
dados_finais = []

# 2. Execução
for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
    paciente_id = row['paciente']
    print("")
    print(paciente_id)
    for tipo in ['a', 'b', 'c']:
        col_name = f'resposta_{tipo}'
        texto_llm = row[col_name]
        linha_score = {"paciente": paciente_id, "resposta": tipo}
        valores_shap, numero_paciente, risco = f.obter_dados_paciente(numero_paciente=paciente_id, path=path)
        # Calcula a soma total para facilitar a escolha da vencedora depois
        is_ok, nota = verificar_coerencia_bertscore(texto_llm, risco, valores_shap)
        linha_score["nota"] = nota
        linha_score["coerente"] = is_ok
        dados_finais.append(linha_score)

# 3. Exportação
df_scores = pd.DataFrame(dados_finais)
df_scores.to_csv("scores_coerencia.csv", index=False)



# --- Resumo Geral ---
total_respostas = len(df_scores)
total_coerentes = df_scores['coerente'].sum()
total_incoerentes = total_respostas - total_coerentes

print(f"--- Extrato de Coerência (BERTScore) ---")
print(f"Total de respostas processadas: {total_respostas}")
print(f"Total de respostas Coerentes: {total_coerentes} ({ (total_coerentes/total_respostas)*100:.2f}%)")
print(f"Total de respostas Incoerentes: {total_incoerentes} ({ (total_incoerentes/total_respostas)*100:.2f}%)")

# --- Análise por Tipo de Resposta (Modelo) ---
# Agrupa por 'resposta' (a, b, c) e calcula métricas
extrato_por_tipo = df_scores.groupby('resposta').agg(
    Media_BERTScore=('nota', 'mean'),
    Qtd_Coerentes=('coerente', 'sum'),
    Total=('coerente', 'count')
)

# Calcula o percentual de aprovação por modelo
extrato_por_tipo['Percentual_Aprovado'] = (extrato_por_tipo['Qtd_Coerentes'] / extrato_por_tipo['Total']) * 100

print("\n--- Desempenho por Modelo ---")
print(extrato_por_tipo[['Media_BERTScore', 'Percentual_Aprovado']])

# Salva o extrato para uso posterior na redação dos resultados
extrato_por_tipo.to_csv("extrato_resumo_coerencia.csv")