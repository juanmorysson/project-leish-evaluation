import os, re
import pandas as pd
import sys
# LLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
# funções
import funcoes as f
import random
import numpy as np


path = ''
temperatura = 0.4
LIMITE = 3088
pacientes_por_rodada = 12
arquivo_txt = "lista_exclusao.txt"
lista_exclusao = []

# Se o arquivo existe, extraímos os números dele
if os.path.exists(arquivo_txt):
    with open(arquivo_txt, "r") as f1:
        conteudo = f1.read().strip()
        if conteudo:
            # Remove os colchetes [ ] e divide pela vírgula
            limpo = conteudo.replace('[', '').replace(']', '').replace(' ', '')
            if limpo: # Garante que não está vazio
                lista_exclusao = [int(n) for n in limpo.split(',')]

def remover_chaves_solitarias(texto):
    # Explicação da Regex:
    # (?<!\{)  -> Garante que não haja uma '{' antes (Negative Lookbehind)
    # \{       -> Procura o caractere '{' literal
    # (?!\{)   -> Garante que não haja uma '{' depois (Negative Lookahead)
    # |        -> OU
    # (?<!\})  -> Garante que não haja uma '}' antes
    # \}       -> Procura o caractere '}' literal
    # (?!\})   -> Garante que não haja uma '}' depois
    
    padrao = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
    
    return re.sub(padrao, '', texto)

def atualizar_exclusao(novo_numero):
    global lista_exclusao
    
    if novo_numero not in lista_exclusao:
        lista_exclusao.append(int(novo_numero))
        #lista_exclusao.sort() # Opcional: mantém os índices em ordem
        
        # Salva a lista inteira no formato solicitado
        with open(arquivo_txt, "w") as f:
            f.write(str(lista_exclusao))
        
        print(f"Número {novo_numero} adicionado. Lista salva: {lista_exclusao}")
    else:
        print(f"O número {novo_numero} já está na lista.")

def avaliador(anotador, resposta, rodada, paciente, diretriz):
    
    llm = ChatOllama(
        model="gemma2:9b", 
        temperature=temperatura,
        max_tokens=None,
        max_retries=1,
    )
    diretriz_tratada = diretriz.replace("{X}", "{{X}}").replace("{risco}", "{{risco}}")

    # Mantenha o sistema focado no papel de auditor
    system_message = (
        "Você é um auditor de IA médica. Sua função é realizar uma auditoria técnica "
        "estrita, comparando a DIRETRIZ ESPERADA com a RESPOSTA ENTREGUE. "
        "Seja conciso, seco e direto. Proibido introduções como 'Aqui está minha análise'."
    )

    # Use delimitadores claros e um formato de saída fixo
    human_template = f"""
    ### CONTEXTO DE AUDITORIA
    DIRETRIZ ATUAL:
    <diretriz_atual>
    {diretriz_tratada}
    </diretriz_atual>

    RESPOSTA GERADA PELO ANOTADOR:
    <resposta_anotador>
    {resposta}
    </resposta_anotador>

    ### TAREFA
    Avalie a conformidade da resposta acima seguindo este checklist:
    1. Atendeu todos os itens solicitado na diretriz atual? (Sim, atendeu todos os itens solicitado na diretriz atual/ Não, não atendeu todos os itens solicitado na diretriz atual)
    2. Analisou todos os valores SHAP que influenciam positivamente para aumentar o risco de morte? (Sim, analisou todos os valores SHAP que influenciam positivamente para aumentar o risco de morte/Não, não analisou todos os valores SHAP que influenciam positivamente para aumentar o risco de morte)
    3. Recomendou ações para o tratamento clínico? (Sim, recomendou ações para o tratamento clínico /Não, não recomendou ações para o tratamento clínico / Parcial, recomendou parcialmente ações para o tratamento clínico)
    4. Apresentou algo que não foi pedido na diretriz atual (alucinações)? (Sim, apresentou alucinações /Não, não apresentou alucinações)

    ### CRÍTICA TÉCNICA (APENAS O QUE PRECISA SER CORRIGIDO)
    """

    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template)
    ])

    # No invoke, passe as variáveis separadas
    chain_gemma = template | llm | StrOutputParser()

    
    # Invoca a chain, passando o prompt formatado
    resposta_critica = chain_gemma.invoke({
        "diretriz": diretriz_tratada, 
        "resposta": resposta
    })

    #print(f"** Crítica da resposta:**\n")
    #print("---------------------------------\n")
    #print(resposta_critica)

    f.add_critica(path, int(rodada), anotador, paciente, resposta_critica)
    return resposta_critica

def amanda(rodada, numero_paciente):
    anotador = 'Amanda'
    llm = ChatOllama(
        model="qwen2.5:7b", 
        temperature=temperatura,
        max_tokens=None,
        max_retries=1,
    )


    # --- 2. Definição do Estado do Grafo (State) ---
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # --- 3. Definição do Nó de Execução (Chain) ---
    def call_llm_node(state: ChatState):
        system_prompt = "Você é um assistente de pesquisa clínica em doenças tropicais, focado na ética da IA e no apoio à decisão médica e com a melhoria da diretriz contida no prompt. Responda de forma curta e baseada em protocolos oficiais"        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # --- 4. Construção e Compilação do Grafo ---
    builder = StateGraph(ChatState)
    builder.add_node("llm_node", call_llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    checkpointer = InMemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    # --- 5. Execução Sequencial com Persistência (Memória) ---
    session_id = "sessao_"+anotador+"_"+str(rodada)
    config = {"configurable": {"thread_id": session_id}}

    # Obtém os dados via OCR automaticamente
    SHAPE_VALUES, numero_paciente, risco = f.obter_dados_paciente(numero_paciente, path)
    #carregar diretriz atual
    diretriz = f.busca_diretriz(rodada, path)

    prompt = "{SHAP_VALUES} \n\n"+diretriz

    prompt_formatado = prompt.format(
        SHAP_VALUES=SHAPE_VALUES,
        X=numero_paciente,
        risco=risco
    )

    # O input é injetado como uma HumanMessage no estado inicial
    resposta_1_state = app.invoke({"messages": [HumanMessage(content=prompt_formatado)]}, config=config)
    resposta_1 = resposta_1_state["messages"][-1].content
    print(f"**{anotador}:**\n")
    #print(f"{prompt_formatado}\n")
    #print("---------------------------------\n")
    #print(f"**IA 1:** {resposta_1}\n")
    #print("---------------------------------\n")
    f.add_resposta(path, rodada, anotador, numero_paciente, resposta_1)

    consideracoes = avaliador(anotador, resposta_1, rodada, numero_paciente,  diretriz)

    # 🔹 Solicita considerações humanas

    prompt = f"""Baseado na resposta anterior 
    <resposta_anterior>
    {resposta_1}
    </resposta_anterior>
    e nas considerações do avaliador sobre sua resposta: 
    <consideracoes>
    {consideracoes}
    </consideracoes>, 
    e nos conceitos de ética na aplicação de IA, e na intenção de melhorar a diretriz contida no prompt,
    sugira melhorias para serem aplicadas no futuro na diretriz atual: 
    <diretriz_atual>
    {diretriz}
    </diretriz_atual>.
    
    OBS: exiba como resposta SOMENTE a nova diretriz sugerida!
    OBS2: assim como na diretriz atual, a nova deve conter OBRIGATORIAMENTE {{X}} e {{risco}} para receber o
    número do paciente e a predição de risco de morte respectivamente."""


    # O input é injetado como uma HumanMessage no estado inicial
    resposta_2_state = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    resposta_2 = resposta_2_state["messages"][-1].content
    f.add_sugestao(path, int(rodada), anotador, numero_paciente, resposta_2)

def beatriz(rodada, numero_paciente):
    anotador = 'Beatriz'
    llm = ChatOllama(
        model="mistral-nemo", 
        temperature=temperatura,
        max_tokens=None,
        max_retries=1,
    )


    # --- 2. Definição do Estado do Grafo (State) ---
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # --- 3. Definição do Nó de Execução (Chain) ---
    def call_llm_node(state: ChatState):
        system_prompt = "Você é um assistente de pesquisa clínica em doenças tropicais, focado na ética da IA e no apoio à decisão médica e com a melhoria da diretriz contida no prompt. Responda de forma curta e baseada em protocolos oficiais"        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # --- 4. Construção e Compilação do Grafo ---
    builder = StateGraph(ChatState)
    builder.add_node("llm_node", call_llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    checkpointer = InMemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    # --- 5. Execução Sequencial com Persistência (Memória) ---
    session_id = "sessao_"+anotador+"_"+str(rodada)
    config = {"configurable": {"thread_id": session_id}}

    # Obtém os dados via OCR automaticamente
    SHAPE_VALUES, numero_paciente, risco = f.obter_dados_paciente(numero_paciente, path)
    #carregar diretriz atual
    diretriz = f.busca_diretriz(rodada, path)

    prompt = "{SHAP_VALUES} \n\n"+diretriz

    prompt_formatado = prompt.format(
        SHAP_VALUES=SHAPE_VALUES,
        X=numero_paciente,
        risco=risco
    )

    # O input é injetado como uma HumanMessage no estado inicial
    resposta_1_state = app.invoke({"messages": [HumanMessage(content=prompt_formatado)]}, config=config)
    resposta_1 = resposta_1_state["messages"][-1].content
    print(f"**{anotador}:**\n")
    #print(f"{prompt_formatado}\n")
    #print("---------------------------------\n")
    #print(f"**IA 1:** {resposta_1}\n")
    #print("---------------------------------\n")
    f.add_resposta(path, rodada, anotador, numero_paciente, resposta_1)

    consideracoes = avaliador(anotador, resposta_1, rodada, numero_paciente,  diretriz)

    # 🔹 Solicita considerações humanas

    prompt = f"""Baseado na resposta anterior 
    <resposta_anterior>
    {resposta_1}
    </resposta_anterior>
    e nas considerações do avaliador sobre sua resposta: 
    <consideracoes>
    {consideracoes}
    </consideracoes>, 
    e nos conceitos de ética na aplicação de IA, e na intenção de melhorar a diretriz contida no prompt,
    sugira melhorias para serem aplicadas no futuro na diretriz atual: 
    <diretriz_atual>
    {diretriz}
    </diretriz_atual>.
    
    OBS: exiba como resposta SOMENTE a nova diretriz sugerida!
    OBS2: assim como na diretriz atual, a nova deve conter OBRIGATORIAMENTE {{X}} e {{risco}} para receber o
    número do paciente e a predição de risco de morte respectivamente."""


    # O input é injetado como uma HumanMessage no estado inicial
    resposta_2_state = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    resposta_2 = resposta_2_state["messages"][-1].content
    f.add_sugestao(path, int(rodada), anotador, numero_paciente, resposta_2)

def carlos(rodada, numero_paciente):
    anotador = 'Carlos'
    llm = ChatOllama(
        #model="llama3.1:8b",
        model="hermes3:8b", 
        temperature=temperatura,
        max_tokens=None,
        max_retries=1,
    )


    # --- 2. Definição do Estado do Grafo (State) ---
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    # --- 3. Definição do Nó de Execução (Chain) ---
    def call_llm_node(state: ChatState):
        system_prompt = "Você é um assistente de pesquisa clínica em doenças tropicais, focado na ética da IA e no apoio à decisão médica e com a melhoria da diretriz contida no prompt. Responda de forma curta e baseada em protocolos oficiais"
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # --- 4. Construção e Compilação do Grafo ---
    builder = StateGraph(ChatState)
    builder.add_node("llm_node", call_llm_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", END)
    checkpointer = InMemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    # --- 5. Execução Sequencial com Persistência (Memória) ---
    session_id = "sessao_"+anotador+"_"+str(rodada)
    config = {"configurable": {"thread_id": session_id}}

    # Obtém os dados via OCR automaticamente
    SHAPE_VALUES, numero_paciente, risco = f.obter_dados_paciente(numero_paciente, path)
    #carregar diretriz atual
    diretriz = f.busca_diretriz(rodada, path)

    prompt = "{SHAP_VALUES} \n\n"+diretriz

    prompt_formatado = prompt.format(
        SHAP_VALUES=SHAPE_VALUES,
        X=numero_paciente,
        risco=risco
    )

    # O input é injetado como uma HumanMessage no estado inicial
    resposta_1_state = app.invoke({"messages": [HumanMessage(content=prompt_formatado)]}, config=config)
    resposta_1 = resposta_1_state["messages"][-1].content
    print(f"**{anotador}:**\n")
    #print(f"{prompt_formatado}\n")
    #print("---------------------------------\n")
    #print(f"**IA 1:** {resposta_1}\n")
    #print("---------------------------------\n")
    f.add_resposta(path, rodada, anotador, numero_paciente, resposta_1)

    consideracoes = avaliador(anotador, resposta_1, rodada, numero_paciente,  diretriz)

    # 🔹 Solicita considerações humanas

    prompt = f"""Baseado na resposta anterior 
    <resposta_anterior>
    {resposta_1}
    </resposta_anterior>
    e nas considerações do avaliador sobre sua resposta: 
    <consideracoes>
    {consideracoes}
    </consideracoes>, 
    e nos conceitos de ética na aplicação de IA, e na intenção de melhorar a diretriz contida no prompt,
    sugira melhorias para serem aplicadas no futuro na diretriz atual: 
    <diretriz_atual>
    {diretriz}
    </diretriz_atual>.
    
    OBS: exiba como resposta SOMENTE a nova diretriz sugerida!
    OBS2: assim como na diretriz atual, a nova deve conter OBRIGATORIAMENTE {{X}} e {{risco}} para receber o
    número do paciente e a predição de risco de morte respectivamente."""


    # O input é injetado como uma HumanMessage no estado inicial
    resposta_2_state = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    resposta_2 = resposta_2_state["messages"][-1].content
    f.add_sugestao(path, int(rodada), anotador, numero_paciente, resposta_2)

def sortear_numero_exclusivo(limite_superior, lista):
    conjunto_total = set(range(limite_superior + 1))
    conjunto_exclusao = set(lista)
    conjunto_validos = conjunto_total - conjunto_exclusao
    lista_validos = list(conjunto_validos)
    if lista_validos:
        numero_sorteado = random.choice(lista_validos)
        #lista_exclusao.append(numero_sorteado)
        return numero_sorteado
    else:
        return None

def medir_similaridades(pacientes):
    medias_pacientes = []
    for paciente in pacientes:
        medias_pacientes.append(f.medir_similaridade(paciente, path))
    media_geral_np = np.mean(medias_pacientes)
    print(f"A Média Geral de similaridade é: {media_geral_np:.4f}")

def ajustar_diretriz(rodada):

    llm = ChatOllama(
        model="phi4", 
        temperature=0.1,
        max_tokens=None,
        max_retries=1,
    )

    sugestoes = f.busca_sugestoes(rodada, path)
    if not sugestoes:
        print("A lista de sugestões está vazia. Não há sugestões válidas para a rodada.")
    else:
        diretriz = f.busca_diretriz(rodada, path)


        # Preparando as sugestões em bloco único para o prompt
        sugestoes_formatadas = [f"<sugestao{i+1}>\n{s}\n</sugestao{i+1}>" for i, s in enumerate(sugestoes)]

        # Unindo tudo em uma única string separada por quebras de linha
        sugestoes_str = "\n".join(sugestoes_formatadas)

        # Aplicando o tratamento das chaves para não quebrar o LangChain
        sugestoes_str_tratada = sugestoes_str.replace("{X}", "{{X}}").replace("{risco}", "{{risco}}")

        sugestoes_str_tratada = remover_chaves_solitarias(sugestoes_str_tratada)
        diretriz_tratada = diretriz.replace("{X}", "{{X}}").replace("{risco}", "{{risco}}")

        # Refinando o System Prompt para ser mais assertivo
        system_message = (
            "Você é um especialista em IA e saúde. Sua única função é reescrever diretrizes de anotação. "
            "Você deve fundir a 'Diretriz Atual' com as 'Sugestões' em um único texto coeso. "
            "REGRAS DE OURO:\n"
            "- Comece com: 'Baseado nos valores shapley acima'\n"
            "- Preserve os marcadores: {{X}} e {{risco}}\n"
            "- Saída: Apenas o texto da nova diretriz, sem explicações."
        )
        
        # Estruturando o prompt humano com delimitadores
        human_message = f"""
        Diretriz atual:
        <diretriz_atual>
        {diretriz_tratada}
        </diretriz_atual>

        Exemplos de diretrizes sugeridas:
        {sugestoes_str_tratada}

        NOVA DIRETRIZ REFINADA:
        - Comece com: 'Baseado nos valores shapley acima'

        """
        human_message.__add__("- Preserve os marcadores: {{X}} e {{risco}}")


        template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

        print("PROMPT::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print(human_message)

        chain = template | llm | StrOutputParser()

        # Invoca a chain, passando o prompt formatado
        resposta = chain.invoke({"prompt": human_message})

        #print("\n--- Resposta do Gemini ---")
        #print(resposta)

        f.add_diretriz(path, rodada, resposta)


def rodar(rodada):
    pacientes = []
    for i in range(0, pacientes_por_rodada):
        numero_paciente = int(sortear_numero_exclusivo(LIMITE, lista_exclusao))
        pacientes.append(numero_paciente)
        atualizar_exclusao(numero_paciente)
        print("Rodada: "+str(rodada))
        print("Paciente: "+str(numero_paciente))
        amanda(rodada, numero_paciente)
        print("Rodada: "+str(rodada))
        print("Paciente: "+str(numero_paciente))
        beatriz(rodada, numero_paciente)
        print("Rodada: "+str(rodada))
        print("Paciente: "+str(numero_paciente))
        carlos(rodada, numero_paciente)
    return pacientes
#for rodada in range(0,rodadas):
rodada = int(input("Digite o numero da rodada: "))

pacientes = rodar(rodada)
#pacientes = [2554,1723,259,2381]
medir_similaridades(pacientes)

ajustar_diretriz(rodada)

