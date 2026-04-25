import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def gerar_relatorio_vencedores(scores_path, responses_path):
    # 1. Carregar dados
    scores_df = pd.read_csv(scores_path)
    resp_df = pd.read_csv(responses_path)
    
    # Inicializar modelo de embeddings
    print("Carregando modelo semantic-BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Identificar a maior nota técnica por paciente
    max_scores = scores_df.groupby('paciente')['nota_total'].transform(max)
    potenciais_vencedores = scores_df[scores_df['nota_total'] == max_scores].copy()
    
    final_results = []
    
    # 3. Processar cada paciente para eleger a melhor resposta (texto)
    for paciente, group in potenciais_vencedores.groupby('paciente'):
        # Localiza a linha do paciente onde as respostas a, b, c estão juntas
        row_resp = resp_df[resp_df['paciente'] == paciente].iloc[0]
        
        # Mapeamento para acessar o texto correto baseado no ID 'a', 'b' ou 'c'
        mapa_textos = {
            'a': row_resp['resposta_a'],
            'b': row_resp['resposta_b'],
            'c': row_resp['resposta_c']
        }
        
        if len(group) == 1:
            # Vencedor único: pegamos o ID e buscamos o texto no mapa
            id_vencedor = group.iloc[0]['resposta']
            tipo_vitoria = 'Julgamento direto'
        else:
            # Empate técnico: Critério de proximidade do Centroide via Embeddings
            print(f"Desempatando paciente {paciente} pelo centroide...")
            tipo_vitoria = 'Desempate por Centroide'
            # Vetores de todas as opções para criar o centroide de referência
            textos_base = [mapa_textos['a'], mapa_textos['b'], mapa_textos['c']]
            embeddings_base = model.encode(textos_base)
            centroid = np.mean(embeddings_base, axis=0).reshape(1, -1)
            
            # Vetores apenas das respostas que empataram na nota máxima
            ids_empatados = group['resposta'].tolist()
            textos_empatados = [mapa_textos[idx] for idx in ids_empatados]
            embeddings_empatados = model.encode(textos_empatados)
            
            # Cálculo de Similaridade (Maior valor = Mais próximo do centroide)
            similaridades = cosine_similarity(embeddings_empatados, centroid).flatten()
            id_vencedor = ids_empatados[np.argmax(similaridades)]
        
        # Guardar apenas o ID do paciente e o CONTEÚDO da resposta
        final_results.append({
            'paciente_id': paciente,
            'melhor_resposta': mapa_textos[id_vencedor],
            'modelo_vencedor': id_vencedor,  # Coluna nova 1
            'tipo_vitoria': tipo_vitoria    # Coluna nova 2
        })
        if paciente % 50 == 0:
            df_vencedores = pd.DataFrame(final_results)
            df_vencedores.to_csv("vencedores.csv", index=False)
            
    # 4. Criar DataFrame final simplificado
    df_vencedores = pd.DataFrame(final_results)
    df_vencedores.to_csv('vencedores.csv', index=False, sep=';', encoding='utf-8-sig')
    
    print(f"Sucesso! {len(df_vencedores)} pacientes processados.")
    return df_vencedores

# Execução
gerar_relatorio_vencedores('scores_judge.csv', 'respostas_melhor_diretriz_corrigido.csv')