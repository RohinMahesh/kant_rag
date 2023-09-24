import streamlit as st
from philosophy_qa_bot.modeling.rag_model import PhilosophyRAG

st.title("Philosophy QA Bot")

with st.form("qa_bot_form"):
    question = st.text_area("Please enter your question:")
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = PhilosophyRAG(question=question)
        st.info(response)
