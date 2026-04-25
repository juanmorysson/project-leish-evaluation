import pandas as pd
import ollama
import re
from tqdm import tqdm
import logging
import warnings

# Silencia logs de detecção de encoding (muito comum em bibliotecas como chardet ou charset_normalizer)
logging.getLogger('charset_normalizer').setLevel(logging.ERROR)
logging.getLogger('chardet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

path = "../ciclo_diretriz/"

# Configuração das 11 Perguntas e suas Escalas
PERGUNTAS = {
    "nota1":  {"pergunta": "O texto identifica corretamente as 4 principais features de maior magnitude do SHAP?", "escala": (1, 5)},
    "nota2":  {"pergunta": "A explicação descreve corretamente a direção do impacto (positivo/negativo)?", "escala": (1, 5)},
    "nota3":  {"pergunta": "A terminologia (peso, influência) é consistente com a magnitude visual do SHAP?", "escala": (1, 5)},
    "nota4":  {"pergunta": "Segue a estrutura formal de laudo clínico (intro, análise, conclusão)?", "escala": (1, 5)},
    "nota5":  {"pergunta": "Cita explicitamente valor base, soma dos impactos e probabilidade final?", "escala": (1, 5)},
    "nota6":  {"pergunta": "Integra achados laboratoriais com a fisiopatologia da Leishmaniose Visceral?", "escala": (1, 5)},
    "nota7":  {"pergunta": "Invenção de correlações médicas inexistentes ou contraditórias (Alucinação)?", "escala": (0, 1)}, # 1 passa, 0 falha
    "nota8":  {"pergunta": "Menção a variáveis não presentes no gráfico SHAP (Alucinação)?", "escala": (0, 1)},
    "nota9":  {"pergunta": "Atribuição de grande impacto a variáveis irrelevantes (Alucinação)?", "escala": (0, 1)},
    "nota10": {"pergunta": "Conduta clínica utiliza terminologia médica adequada (vigilância, ajuste de dose)?", "escala": (1, 5)},
    "nota11": {"pergunta": "O insight é logicamente derivado das features de maior impacto no SHAP?", "escala": (1, 5)}
}

MODEL_NAME = "llama3:70b"

def extrair_nota_estrita(texto, min_val, max_val):
    """Extrai apenas o número final, ignorando o raciocínio <think>."""
    texto_limpo = re.sub(r'<think>.*?</think>', '', texto, flags=re.DOTALL)
    numeros = re.findall(r'\b\d+\b', texto_limpo)
    if numeros:
        nota = int(numeros[-1]) # Pega o último número para evitar pegar números citados na pergunta
        if min_val <= nota <= max_val:
            return nota
    return None

def avaliar_quesito(resposta_texto, info_pergunta, ip, t, n):
    """Pergunta ao modelo uma questão específica por vez."""
    min_v, max_v = info_pergunta["escala"]
    # Instrução de sistema para travar o comportamento do modelo
    prompt = f"""
    [CONTEXTO] Avalie uma explicação técnica de um gráfico SHAP sobre risco de óbito em Leishmaniose Visceral.
    [PERGUNTA] {info_pergunta['pergunta']}
    [ESCALA] {min_v} a {max_v}
    [RESPOSTA A AVALIAR] \"\"\"{resposta_texto}\"\"\" [FIM DO TEXTO]
    
    Responda apenas com o número da nota final.
    Regra de Ouro: Responda apenas o número da nota em Português. Não explique.
    Nota:"""
    
    
    for _ in range(3): # Retry em caso de nota fora da escala
        res = ollama.generate(
            model=MODEL_NAME, 
            think=False,
            prompt=prompt, 
            options={
                "temperature": 2,      # Garante maior consistência nas notas
                "top_k": 1,
                "top_p": 0.1,
                #"num_predict": 10      # Limita a resposta para ser curta (apenas a nota)
                "num_ctx": 4096,
                "stop": ["\n", "Score:", "Nota:"]
                }
        )
        conteudo_resposta = res['response'].strip()
        nota = extrair_nota_estrita(conteudo_resposta, min_v, max_v)
        if nota is not None:
            return nota
        else:
            print("repetiu: "+str(ip)+"-"+t+"-"+str(n))
            #print(conteudo_resposta)
            #print("XXXXXXXXXXXXXXXXXX")
            #print(prompt)

    return 0 # Default para falha persistente

# 1. Preparação
df_input = pd.read_csv("respostas_melhor_diretriz_corrigido.csv") # Deve conter paciente, resposta_a, resposta_b, resposta_c
dados_finais = []

df_coerencias = pd.read_csv("scores_coerencia.csv")

# 2. Execução
for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
    paciente_id = row['paciente']
    #print("")
    #print(paciente_id)
    for tipo in ['a', 'b', 'c']:
        #print(tipo)
        #verificar se é coerente
        filtro = (df_coerencias['paciente'] == paciente_id) & (df_coerencias['resposta'] == tipo)
        resultado = df_coerencias[filtro]
        coerente = bool(resultado['coerente'].values[0])
        if coerente:
            col_name = f'resposta_{tipo}'
            texto_llm = row[col_name]
            
            linha_score = {"paciente": paciente_id, "resposta": tipo}
            
            for id_nota, info in PERGUNTAS.items():
                #print(id_nota)
                nota = avaliar_quesito(texto_llm, info, paciente_id, tipo, id_nota)
                if id_nota in ["nota7","nota8","nota9"]:
                    if nota == 1:
                        nota = 0
                    elif nota == 0:
                        nota = 1
                linha_score[id_nota] = nota
            
            # Calcula a soma total para facilitar a escolha da vencedora depois
            linha_score["nota_total"] = sum(linha_score[n] for n in PERGUNTAS.keys())
            dados_finais.append(linha_score)
    if paciente_id % 50 == 0:
        df_scores = pd.DataFrame(dados_finais)
        df_scores.to_csv("scores_judge.csv", index=False)
# 3. Exportação
df_scores = pd.DataFrame(dados_finais)
df_scores.to_csv("scores_judge.csv", index=False)
