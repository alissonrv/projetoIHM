import streamlit as st

st.title("Deletar Chat")
chats = list(st.session_state.chats.keys())

if not chats:
    st.info("Nenhum chat para deletar.")
    st.stop()

a_deletar = st.selectbox("Escolha o chat para deletar", chats)
if st.button("Deletar"):
    st.session_state.chats.pop(a_deletar)
    if st.session_state.current_chat == a_deletar:
        st.session_state.current_chat = None
    st.success(f"Chat “{a_deletar}” deletado.")
    st.rerun()

st.set_page_config(page_title="Deletar Chat", page_icon=":material/delete:")