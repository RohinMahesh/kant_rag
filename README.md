# Philosophy QA Bot

The data source used for this QA bot is The Groundwork to the Metaphysics of Morals by Immanuel Kant. In generating a knowledge base to serve as context in the QA Bot, the challenge is common themes can have conflicting view points based on the branch of philosophy.  For example, a prompt for "What is justice?" can have varying answers based on the school of thought (Stoic, Socratic, etc.). In an attempt to reduce potential hallucinations, this corpus will serve as the context for question and answering.

The QA bot was created through the following process:
    1. Loading of JSON file containing corpus for knowledge base
    2. Creation of embeddings using distilled LLMs from HuggingFace 
    3. Creation of FAISS index file to serve downstream similarity searches
    4. Creation of context given results of similarity search via FAISS
    5. Creation of prompt using LangChain
    6. Initialization of OpenAI client
    7. Creation of LLMChain using LangChain
    8. Execution of LLMChain to serve response for a given question