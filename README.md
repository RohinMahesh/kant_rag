# Philosophy Retrieval Augmented Generation (RAG)

The documents used for this QA bot is based on works from The Groundwork to the Metaphysics of Morals by Immanuel Kant to create our knowledge base. We then perform Retrieval Augmented Generation, which leverages our knowledge base to better answer a given question.

In generating a knowledge base, the challenge in this domain is that common themes can have conflicting viewpoints based on the branch of philosophy.  For example, a prompt for "What is justice?" can have varying answers based on the school of thought (Stoic, Socratic, etc.). In an attempt to reduce potential hallucinations, only the aforementioned documents will be used to create a knowledge base for our downstream question and answering.

The QA bot was created through the following process: 
- Loading of JSON file containing the corpus for our knowledge base
- Conversion of the text/metadata into Documents and creation of FAISS index
- Creation of LangChain prompt
- Initialization of OpenAI client and LangChain retriever
- Creation of LangChain RetrievalQA chain
- Execution of RetrievalQA to serve response for a given question

This Philosophy QA bot is run as a Streamlit application, which can be found in applications/app.py. For reproducibility, the underlying FAISS index for RAG is made available under data/faiss_index. 
