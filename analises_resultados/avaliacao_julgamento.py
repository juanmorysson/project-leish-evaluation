import pandas as pd

# 1. Carregar os dados
df = pd.read_csv('scores_judge.csv')

# 2. Definir os grupos de colunas por critério
criteria_map = {
    'Fidelity': ['nota1', 'nota2', 'nota3'],
    'Completeness': ['nota4', 'nota5', 'nota6'],
    'Hallucination': ['nota7', 'nota8', 'nota9'],
    'Actionable Insights': ['nota10', 'nota11']
}

# 3. Calcular a média de cada critério para cada linha
for criterion, columns in criteria_map.items():
    df[criterion] = df[columns].mean(axis=1)

# 4. Agrupar por tipo de resposta e calcular a média geral
stats = df.groupby('resposta')[list(criteria_map.keys())].mean()

# 5. Exibir resultados
print("Média por Critério:")
print(stats)

print("\nMelhor resposta para cada critério:")
for crit in criteria_map.keys():
    melhor_nota = stats[crit].max()
    vencedores = stats[stats[crit] == melhor_nota].index.tolist()
    print(f"- {crit}: {', '.join(vencedores)} (Nota: {melhor_nota})")



import matplotlib.pyplot as plt

# 1. Transpor os dados para que os Critérios fiquem no eixo X e as Respostas na Legenda
plot_data = stats.T

# 2. Criar o gráfico de barras
# Definimos as cores para manter a consistência visual
ax = plot_data.plot(kind='bar', figsize=(10, 4), width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# 3. Customização estética para o formato acadêmico (estilo Springer/LNCS)
#plt.title('Comparação de Desempenho: Respostas A, B e C', fontsize=14, fontweight='bold')
plt.ylabel('Average Score', fontsize=12)
plt.xlabel('Evaluation Criteria', fontsize=12)
plt.xticks(rotation=0)  # Deixa os nomes dos critérios na horizontal
plt.legend(
    title='Responses from', 
    labels=['Qwen 2.5 7B', 'Mistral-Nemo','Hermes 3 8B'],
    loc='lower center',       # Define o ponto de referência da legenda
    bbox_to_anchor=(0.62, 0.6), # (x, y) - 0.62 centraliza sobre Hallucination, 1 coloca acima do gráfico
)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 4. Ajustar o limite do eixo Y (geralmente a escala é de 1 a 5)
plt.ylim(0, 5.2) 

# 5. Adicionar os valores exatos sobre cada barra (Rótulos de dados)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 8), 
                textcoords='offset points',
                fontsize=9,
                fontweight='bold')

# 6. Finalizar e salvar
plt.tight_layout()
plt.savefig('resultado_avaliacao_llm.png', dpi=300) # Salva em alta resolução para o artigo
#plt.show()