# Q-A-tool
This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using FAISS for similarity search, Sentence-BERT embeddings, and Flan-T5 for summarization and Q&A.

The system processes a PDF document, splits it into chunks, embeds them into a FAISS index (index_pdf.py) , and later answers user questions by retrieving the most relevant chunks and generating answers (question_answering.py).
