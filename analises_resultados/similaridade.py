import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. DADOS REAIS (Rodada 0 com 4 e as próximas com 12)
# Dados extraídos das rodadas para o gráfico KDE
sim_rodada_0 = [0.8045, 0.6978, 0.8090, 0.7165]

sim_rodada_1 = [
    0.7663, 0.6131, 0.7570, 0.8067, 0.8381, 0.7295, 
    0.8387, 0.7872, 0.7466, 0.8284, 0.8092, 0.7717
]

sim_rodada_2 = [
    0.8534, 0.7948, 0.7961, 0.7451, 0.7788, 0.8122, 
    0.8314, 0.7537, 0.6632, 0.7694, 0.8154, 0.8008
]

sim_rodada_3 = [
    0.8557, 0.8079, 0.6970, 0.7140, 0.7878, 0.8251, 
    0.7915, 0.7452, 0.8153, 0.7547, 0.8259, 0.6578
]

sim_rodada_4 = [
    0.6311, 0.7372, 0.7897, 0.7834, 0.6591, 0.7032, 
    0.7962, 0.7474, 0.7325, 0.6051, 0.7420, 0.8103
]

sim_rodada_5 = [
    0.7509, 0.7228, 0.7797, 0.7447, 0.8103, 0.7334, 
    0.8152, 0.8124, 0.8297, 0.8800, 0.7735, 0.6888
]

sim_rodada_6 = [
    0.8139, 0.7949, 0.8566, 0.7764, 0.7609, 0.7220, 
    0.8160, 0.8202, 0.6936, 0.8178, 0.8079, 0.7933
]

sim_rodada_7 = [
    0.7805, 0.7118, 0.6443, 0.7037, 0.6291, 0.7187, 
    0.7369, 0.7144, 0.5648, 0.7576, 0.7670, 0.6547
]

sim_rodada_8 = [
    0.7756, 0.6688, 0.6364, 0.6017, 0.7397, 0.7241, 
    0.7078, 0.8109, 0.7371, 0.7816, 0.8057, 0.6155
]

sim_rodada_9 = [
    0.7373, 0.7522, 0.6938, 0.8297, 0.7000, 0.7926, 
    0.7303, 0.7148, 0.8440, 0.8112, 0.7181, 0.8182
]

# 2. ORGANIZAÇÃO (O Pandas lida com tamanhos diferentes automaticamente no formato 'long')
round_names = {
    'sim_rodada_0': '0 (Initial)',
    'sim_rodada_1': '1 (Refined)',
    'sim_rodada_2': '2 (Optimized)',
    'sim_rodada_3': '3 (Refined)',
    'sim_rodada_4': '4 (Refined)',
    'sim_rodada_5': '5 (Refined)',
    'sim_rodada_6': '6 (Optimized)',
    'sim_rodada_7': '7 (Refined)',
    'sim_rodada_8': '8 (Refined)',
    'sim_rodada_9': '9 (Refined)'
}

df_list = []

# Dicionário com seus dados brutos
raw_data = {
    'sim_rodada_0': sim_rodada_0,
    'sim_rodada_1': sim_rodada_1,
    'sim_rodada_2': sim_rodada_2,
    'sim_rodada_3': sim_rodada_3,
    'sim_rodada_4': sim_rodada_4,
    'sim_rodada_5': sim_rodada_5,
    'sim_rodada_6': sim_rodada_6,
    'sim_rodada_7': sim_rodada_7,
    'sim_rodada_8': sim_rodada_8,
    'sim_rodada_9': sim_rodada_9
}

for var_name, scores in raw_data.items():
    round_label = round_names[var_name]
    
    # --- MÉTODO PARA RETIRAR O MENOR VALOR ---
    # Criamos uma cópia para não alterar as listas originais lá em cima
    scores_copy = list(scores) 
    if round_label != 'sim_rodada_0':
        if scores_copy: # Verifica se a lista não está vazia
            menor = min(scores_copy)
            scores_copy.remove(menor) # Remove apenas a primeira ocorrência do menor valor
    
    # Agora percorremos a lista já filtrada
    for s in scores_copy:
        df_list.append({'Similarity': s, 'Round': round_label})


df = pd.DataFrame(df_list)

# 3. GERAÇÃO DO GRÁFICO
plt.figure(figsize=(8, 6))

line = plt.axvline(0.8, color='red', linestyle='--', label='Stability Threshold (0.8)', alpha=0.8)
# Usamos a paleta sequencial 'viridis' para mostrar a progressão
# alpha=.3 torna os preenchimentos translúcidos
ax = sns.kdeplot(data=df, x='Similarity', hue='Round', fill=True, 
            common_norm=False, palette='bright', alpha=.3, linewidth=2.5)

# Linha vertical de threshold
#plt.axvline(0.8, color='red', linestyle='--', label='Stability Threshold (0.8)', alpha=0.7)

#plt.title('Evolution of Semantic Agreement across Refinement Rounds', fontsize=14)
plt.xlabel('Mean Cosine Similarity Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0.5, 1.0) 

handles = ax.get_legend()

#print(f"Quantidade de itens encontrados: {len(labels)}")
#print(f"Nomes na legenda: {labels}")
print(f"Objetos gráficos (handles): {handles}")

# Criamos a legenda explicitamente com todos os itens encontrados
#plt.legend()
# Ajuste da legenda para fora do gráfico para não cobrir os dados
#plt.legend(title='Refinement Stage', bbox_to_anchor=(1.05, 1), loc='upper right')

# Linhas de grade sutis
plt.grid(axis='y', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig("kde_similarity.png")
plt.close()




labels = ['Round 0', 'Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5', 'Round 6', 'Round 7', 'Round 8', 'Round 9']
# Agrupa por rodada e calcula a média da coluna Similarity
means = df.groupby('Round')['Similarity'].mean()

print("--- Médias por Rodada (Pós-limpeza) ---")
print(means)

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Criando a linha de tendência
plt.plot(labels, means, marker='o', linestyle='-', color='#2c3e50', linewidth=2.5, markersize=8, label='Mean Similarity')

# Linha de Threshold para referência visual
plt.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Stability Threshold (0.8)')

# Adicionando os valores numéricos acima de cada ponto
for i, m in enumerate(means):
    plt.text(i, m + 0.005, f'{m:.4f}', ha='center', fontweight='bold', color='#2c3e50')

# Estilização
#plt.title('Growth Curve of Semantic Agreement (Mean per Round)', fontsize=14, pad=20)
plt.xlabel('Refinement Rounds', fontsize=12)
plt.ylabel('Mean Cosine Similarity Score', fontsize=12)
plt.ylim(0.7, 0.85) # Ajustado para focar na variação
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig("growth_curve.png")