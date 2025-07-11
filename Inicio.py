#Dependências
# pip install pandas
# pip install langchain_community
# pip install -qU langchain-community unstructured openpyxl
# pip install -qU langchain-huggingface
# pip install -qU sentence-transformers
# pip install -qU langchain-community faiss-cpu
# pip install -qU langchain-groq
# pip install streamlit

import pandas as pd
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_pinecone import PineconeEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import streamlit as st

df = pd.read_excel("BaremaSI.xlsx")

docs = []
for _, row in df.iterrows():
    conteudo = (
        f"Natureza: {row['Natureza']}\n"
        f"Classificação: {row['Classificação']}\n"
        f"Atividades: {row['Atividades']}\n"
        f"Pontuação: {row['Pontuação']}\n"
        f"Comprovação: {row['Comprovação']}"
    )
    metadata = {
        "natureza": row['Natureza'],
        "classificacao": row['Classificação'],
        "atividade": row['Atividades'],
        "pontuacao": row['Pontuação'],
        "comprovacao": row['Comprovação']
    }
    docs.append(Document(page_content=conteudo, metadata=metadata))

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatGroq(
    # model="deepseek-r1-distill-llama-70b",
    # model="gemma2-9b-it",
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

RAG_TEMPLATE = """
Você é um assistente especializado em responder perguntas sobre Horas Complementares do curso de Sistemas de Informação.
– Utilize apenas os trechos de contexto recuperados pelo sistema de RAG.
– Cada contexto vem associado a metadados: natureza, classificação, atividades, pontuação e comprovação.
– Antes de responder, filtre mentalmente os trechos pelo metadado mais relevante à pergunta (por exemplo, “natureza: Extensão” ou “classificação: Ensino”).
– Retorne a resposta de forma clara e objetiva. Ao responder diga "de acordo com as informações encontradas..."
– Pode fazer uma pause de até 5 segundos para pensar melhor antes de responder a pergunta.
– Se não encontrar a informação, responda que não entendeu muito bem a pergunta e peça pro usuário incluir mais detalhes. Você pode dar sugestões com informações que achar relevante.
- Se a pessoa não fizer uma pergunta específica sobre o barema, converse com ela normalmente. Se não souber o que responder apenas diga que não sabe.

<context>
{context}
</context>

Pergunta: {prompt}

Resposta:
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
chain = LLMChain(prompt=rag_prompt, llm=llm)

st.write("# Bem vindo ao SIChat! 👋🤖")

st.markdown(
    """
    Este aplicativo é um protótipo de chatbot construído com a finalidade de ajudar os alunos do
    curso de Sistemas de Informação da UAST com dúvidas sobre horas complementares. As informações do aplicativo se baseiam no Barema
    do curso, que se encontra nesse link 👉 [BaremaSI](https://drive.google.com/file/d/1pnCOoWsIIywgsL0OZFH56RgeC6PvAi70/view?usp=sharing)
    Obrigado!
    """
)


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

chat_names = list(st.session_state.chats.keys())
selected = st.selectbox("Selecione um chat", [" "] + chat_names)
if selected in chat_names:
    st.session_state.current_chat = selected

if not st.session_state.current_chat:
    st.info("Nenhum chat selecionado. Crie um novo em 'Novo Chat'")
    st.stop()

messages = st.session_state.chats[st.session_state.current_chat]
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Digite sua mensagem")
if prompt:
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    docs = vectorstore.similarity_search(prompt, k=4)
    context = [d.page_content for d in docs]
    res = chain.invoke({"context": context, "prompt": prompt})
    assistant_text = res["text"]

    messages.append({"role": "assistant", "content": assistant_text})
    st.chat_message("assistant").markdown(assistant_text)

st.set_page_config(page_title="Início", page_icon=":material/home:")
