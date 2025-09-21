# Q-A-tool
This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using FAISS for similarity search, Sentence-BERT embeddings, and Flan-T5 for summarization and Q&A.

The system processes a PDF document, splits it into chunks, embeds them into a FAISS index (index_pdf.py) , and later answers user questions by retrieving the most relevant chunks and generating answers (question_answering.py).
## 📂 Project Structure

### `index_pdf.py`
- Extracts text from the input PDF.  
- Splits text into overlapping chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`).  
- Encodes chunks into embeddings using **Sentence-BERT (`all-mpnet-base-v2`)**.  
- Normalizes embeddings and stores them in a **FAISS index**.  
- Saves both the FAISS index (`faiss_index.index`) and metadata (`metas.pkl`) for later retrieval.  

👉 Run this **once** to preprocess your document.  

---

### `qa_flan_t5.py`
- Loads the FAISS index and metadata (runs `index_pdf.py` automatically if no index exists).  
- Uses **retrieval** to find the `TOP_K` most relevant chunks for a given question.  
- Summarizes retrieved chunks with **Flan-T5** to keep the context concise.  
- Builds a prompt including:
  - The **user’s question**  
  - The **summaries of top chunks**  
- Passes the prompt to Flan-T5 to generate a final answer.  
- Provides an **interactive CLI** for asking questions.  

---

## Usage

1. **Index your PDF** (first time only):  
   ```bash
   python index_pdf.py
2. **Run the Q&A loop**
   ```bash
   python question_answering.py

## Examples 
1. Question :
   <img width="1173" height="33" alt="Q3" src="https://github.com/user-attachments/assets/88606b7c-1d8e-45ad-8848-de8fc43c316d" />
Answer :
<img width="1552" height="164" alt="Capture d'écran 2025-09-21 154350" src="https://github.com/user-attachments/assets/81160b45-3d58-44c3-815b-b8b544d3ba46" />

3. Question :
<img width="1055" height="43" alt="Q1" src="https://github.com/user-attachments/assets/00e87a80-2d2a-40af-9f5e-20fa3f49ced6" />

Answer : 
  <img width="1538" height="156" alt="Answer1" src="https://github.com/user-attachments/assets/ccdd36cb-8199-4f98-8530-3e8e7f9d21df" />
3. Question :
<img width="1178" height="30" alt="Q2" src="https://github.com/user-attachments/assets/6afc0c8f-80bf-4394-8e91-f02289c4c5fd" />

Answer :
<img width="1547" height="137" alt="A2" src="https://github.com/user-attachments/assets/36afb2dd-5aee-47c9-836d-62cf5d381163" />


## Limitations 
- Mixed language output: Flan-T5 sometimes produces English instead of French.

- Hallucinations: Model may invent information not present in the text.

- Context truncation: Long questions + many chunks may exceed input length (currently handled by truncation).

- Performance: Indexing large PDFs and loading models can be slow on CPU.
