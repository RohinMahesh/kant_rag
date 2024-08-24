import streamlit as st
from kant_rag.modeling.rag_model import KantRAG

st.title("Kant QA")

with st.form("qa_bot_form"):
    question = st.text_area("Please enter your question:")
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = KantRAG(question=question)
        st.info(response)
