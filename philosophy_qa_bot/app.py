import streamlit as st
from philosophy_qa_bot.philosophy_qa_bot.philosophy_qa_bot import PhilosophyQABot

st.title("Philosophy QA Bot")

with st.form("qa_bot_form"):
    question = st.text_area("Please enter your question:")
    submitted = st.form_submit_button("Submit")
    response = PhilosophyQABot(question=question)
    st.info(response)
