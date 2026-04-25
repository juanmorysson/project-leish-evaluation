import pandas as pd
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
import funcoes as f
from typing import TypedDict, Annotated

tempo_delay = 30
path = ''
temperatura = 0.4
rodada = int(input("Digite o numero da rodada: "))
df = pd.read_csv("dataset_processado_437.csv", sep=';')
tamanho = len(df)

try:
    respostas_melhor_diretriz = pd.read_csv("respostas_melhor_diretriz.csv", sep=';', encoding='latin-1')
except FileNotFoundError:
    # Se o arquivo não existir, cria o DataFrame do zero
    respostas_melhor_diretriz = pd.DataFrame(columns=["paciente", "resposta_a", "resposta_b", "resposta_c"])
except Exception as e:
    # Caso ocorra outro erro (ex: arquivo corrompido), você fica sabendo o que foi
    print(f"Erro inesperado: {e}")
    respostas_melhor_diretriz = pd.DataFrame(columns=["paciente", "resposta_a", "resposta_b", "resposta_c"])

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
    return resposta_1

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
    return resposta_1    

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
    return resposta_1
  
def salvar_csv(path, rodada, anotador, paciente, resposta):
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


def rodar(rodada):
    for numero_paciente in range(2301, tamanho):
        print("Paciente: "+str(numero_paciente))
        respostas_melhor_diretriz.loc[numero_paciente] = [
            numero_paciente,
            amanda(rodada, numero_paciente),
            beatriz(rodada, numero_paciente),
            carlos(rodada, numero_paciente)
        ]
        if numero_paciente % 50 == 0:
            respostas_melhor_diretriz.to_csv("respostas_melhor_diretriz.csv", sep=';', index=False, encoding='latin-1', errors='ignore')
            time.sleep(tempo_delay)
            print("passou tempo")
    respostas_melhor_diretriz.to_csv("respostas_melhor_diretriz.csv", sep=';', index=False, encoding='latin-1', errors='ignore')
    time.sleep(tempo_delay)
    print("passou tempo")

rodar(rodada)
