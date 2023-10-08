# Retrieval Augmented Generation with LangChain

![Screenshot](retrieval_augmented_generation_with_langchain/docs/images/langchain_banner.png)

# Motivation

Historically, one of the more challenging use cases in the Information Retrieval (IR) domain has been around building solutions to support question and answering. As organizations strive to become digital-first and undergo respective digital transformations, it is imperative to enable services within products to intelligently provide quick and accurate answers to their customers questions. This is crucial in providing a positive customer experience at every touchpoint a customer has with the product. 

In addition to having an ample amount of high-quality labeled data, a key challenge in this area has been around efficient document retrieval methods to generate domain-specific context for downstream language models. Because of this, historically, leading models alone are insufficient to answer several domain-specific questions accurately and reliably. Modern day Large Language Models (LLMs) will require some sort of fine-tuning (i.e., PEFT, LoRA) to provide domain-specific answers.

The specific task we focus on in this repository is a request I have received from several of my former customers - creating a question and answering solution to use existing information around the service's offerings to answer a questions question.

# Background

The documents used for the project is based on works from The Groundwork to the Metaphysics of Morals by Immanuel Kant to create our knowledge base. We then perform Retrieval Augmented Generation, which leverages our knowledge base to provide domain-specific answers.

In generating a knowledge base, the challenge in this domain is that common themes can have conflicting viewpoints based on the branch of philosophy. For example, a prompt for "What is justice?" can have varying answers based on the school of thought (Stoic, Socratic, etc.). In an attempt to reduce potential hallucinations, only the aforementioned documents will be used to create a knowledge base for our downstream question and answering. 

A key consideration in generating this knowledge base would be semantic chunking. There are several strategies for chunking, such as fixed-size chunking, content-aware chunking, and recursive chunking. In this implementation, I utilized content-aware chunking via naïve splitting of paragraphs into documents for downstream consumption.

The QA bot was created through the following process: 
- Loading of JSON file containing the corpus for our knowledge base
- Conversion of the text/metadata into Documents and creation of FAISS index
- Creation of LangChain prompt
- Initialization of OpenAI client and LangChain retriever
- Creation of LangChain RetrievalQA chain
- Execution of RetrievalQA to serve response for a given question

This is run as a Streamlit application, which can be found in applications/app.py. For reproducibility, the underlying FAISS index for RAG is made available under data/faiss_index. 

