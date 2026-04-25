from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ciclo_diretriz import funcoes as f
path = "../ciclo_diretriz/"
# Configuração do Modelo (Gemma)
llm = Ollama(model="gemma3:4b")

def melhorar_primeiro_paragrafo(resposta_original):
    print("alterando")
    # Separa o primeiro parágrafo do restante do texto
    partes = resposta_original.split('\n', 1)
    primeiro_paragrafo = partes[0]
    resto_do_texto = partes[1] if len(partes) > 1 else ""

    # Template focado em transformar "diretriz" em "introdução direta"
    template = """
    Você é um assistente de edição especializado em textos médicos e IA.
    Sua tarefa é REESCREVER apenas o parágrafo de introdução abaixo.
    
    OBJETIVO:
    - O parágrafo deve introduzir a análise de um gráfico SHAP Waterfall para um paciente específico.
    - Remova títulos genéricos como "Diretriz Refined" ou "Manual de Manejo".
    - Seja direto e profissional.
    
    EXEMPLO DE TOM BOM:
    "O gráfico SHAP waterfall para o paciente [ID] com Leishmaniose Visceral apresenta as seguintes características importantes:"

    PARÁGRAFO ORIGINAL PARA REESCREVER:
    {paragrafo_ruim}

    PARÁGRAFO REESCRITO:
    """

    prompt = PromptTemplate(template=template, input_variables=["paragrafo_ruim"])
    chain = prompt | llm | StrOutputParser()

    # 5. Execução
    nova_introducao = chain.invoke({"paragrafo_ruim": primeiro_paragrafo})

    # Retorna o texto reconstruído
    return f"{nova_introducao.strip()}\n\n{resto_do_texto.strip()}"


df_input = pd.read_csv(path+"respostas_melhor_diretriz_sub.csv",encoding='latin-1', sep=';') # Deve conter paciente, resposta_a, resposta_b, resposta_c
dados_finais = pd.DataFrame(columns=df_input.columns)
print("leu")
# 2. Execução
for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
    paciente_id = row['paciente']
    print(paciente_id)
    linha = {"paciente": paciente_id, 
             "resposta_a": melhorar_primeiro_paragrafo(row['resposta_a']),
             "resposta_b": melhorar_primeiro_paragrafo(row['resposta_b']),
             "resposta_c": melhorar_primeiro_paragrafo(row['resposta_c'])}
    linha_df = pd.DataFrame([linha])
    dados_finais = pd.concat([dados_finais, linha_df], ignore_index=True)
    #dados_finais.append(linha)

# 3. Exportação
dados_finais.to_csv("respostas_melhor_diretriz_corrigido_sub.csv", index=False)