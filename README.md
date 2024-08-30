# Kant Retrieval Augmented Generation (RAG)

![Build Status](https://github.com/RohinMahesh/kant_rag/actions/workflows/ci.yml/badge.svg)

![Screenshot](kant_rag/docs/images/langchain_banner.png)

# Motivation

Historically, one of the more challenging use cases in the Information Retrieval (IR) domain has been around building solutions to support question answering. As organizations strive to become digital-first and undergo respective digital transformations, it is imperative to enable services within products to intelligently provide quick and accurate answers to key product related questions. This is crucial in providing a positive customer experience at every touchpoint a customer has with the product. 

In addition to having an ample amount of high-quality labeled data, a key challenge in this area has been around efficient document retrieval methods to generate domain-specific context for downstream language models. Because of this, historically, leading models alone are insufficient to answer several domain-specific questions accurately and reliably. Modern day Large Language Models (LLMs) may require fine-tuning to provide domain-specific answers.

The specific task we focus on in this repository is a request I have received from several of my former customers - creating a question answering solution that uses existing information around the service's offerings to answer a questions question.

# Background

The documents used for the project is based on works from The Groundwork to the Metaphysics of Morals by Immanuel Kant to create our knowledge base. We then perform Retrieval Augmented Generation, which leverages our knowledge base to provide domain-specific answers.

In generating a knowledge base, the challenge in this domain is that common themes can have conflicting viewpoints based on the branch of philosophy. For example, a prompt for "What is justice?" can have varying answers based on the school of thought (Stoic, Socratic, etc.). In an attempt to reduce potential hallucinations, only the aforementioned documents will be used to create a knowledge base for our downstream question and answering. 

# Approach

A key consideration in generating this knowledge base would be semantic chunking. There are several strategies for chunking, such as fixed-size chunking, content-aware chunking, and recursive chunking. In this implementation, I utilized content-aware chunking via naïve splitting of paragraphs into documents for downstream consumption.

The QA bot was created through the following process: 
- Loading of JSON file containing the corpus for our knowledge base
- Conversion of the text/metadata into Documents and creation of FAISS index
- Creation of LangChain prompt
- Initialization of OpenAI client and LangChain retriever
- Creation of LangChain RetrievalQA chain
- Execution of RetrievalQA to serve response for a given question

This is run as a Streamlit application, which can be found in applications/app.py. For reproducibility, the underlying FAISS index for RAG is made available under data/faiss_index. 

# Evaluation

To evaluate our RAG, Ragas is used to evaluate both the generation and the retrieval of our RAG. 
- Context relevancy
- Context recall
- Answer relevancy
- Faithfulness

These 4 metrics together are used to calculate the "Ragas Score", which can be calculated by running the following file: 

evaluation/evaluate.py

It is recommended to establish Red-Amber-Green thresholds based on this score and integrate this into the LLMOps stack to trigger downstream actions, such as recreating the embeddings and index files.

# Responsible AI

In order to promote Resonsible AI standards, Giskard is used to scan for the following:
- Injection attacks
- Hallucination and misinformation
- Harmful content generation
- Stereotypes
- Information disclosure

It is imperative that the responses from the RAG does not violate any of the Responsible AI risks above that are scanned by Giskard. To perform this scan, integrate the following file into your LLMOps stack: 

responsible_ai_scan/scan.py

# Disclaimers

- With the current version of Ragas, it is a known issue that the environment variable for "OPENAI_API_KEY" must be set before importing ragas related libraries or you will face errors with OpenAI.