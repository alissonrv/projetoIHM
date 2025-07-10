import streamlit as st

st.title("Novo Chat")
st.write("Dê um nome para o novo chat:")

nome = st.text_input("Nome do chat")
if st.button("Criar chat"):
    if not nome:
        st.error("Por favor, insira um nome.")
    elif nome in st.session_state.chats:
        st.error("Já existe um chat com esse nome.")
    else:
        st.session_state.chats[nome] = []
        st.session_state.current_chat = nome
        st.success(f"Chat “{nome}” criado e selecionado!")
        st.rerun()

st.set_page_config(page_title="Novo Chat", page_icon=":material/add_circle:")