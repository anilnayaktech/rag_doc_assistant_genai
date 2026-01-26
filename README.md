# 🧠 GenAI RAG Chatbot (LangChain + HuggingFace + FAISS)

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions **only from provided documents**, built using **LangChain, Hugging Face models, FAISS vector search,** and **Streamlit**.

---

## 🚀 Features
- 🔍 Semantic search using FAISS
- 🧠 Hugging Face LLM (FLAN-T5)
- 📄 Answers grounded in documents
- 🔐 Input safety filtering
- 📚 Shows source documents for transparency
- 🖥️ Simple Streamlit web UI

---

## 🚀 Live Demo
Check out the deployed app here: [GenAI RAG Chatbot](https://akn-rag-doc-assistant-genai.streamlit.app/)

---
## 🏗️ Project Architecture
```text

User Question
     ↓
Safety Check
     ↓
Retriever (FAISS)
     ↓
Relevant Context
     ↓
LLM (Hugging Face)
     ↓
Answer + Source Documents

```
## 📁 Project Structure
```text

rag_doc_assistant_genai/
│
├── scripts/
│   ├── embeddings.py        # Vector store (FAISS)
│   ├── rag_pipeline.py      # RAG chain logic
│   ├── safety.py            # Input safety checks
│   ├── evaluation.py        # RAG evaluation
│   └── finetune.py          # Fine-tuning experiments
│
├── data/
│   └── sample.txt           # Knowledge source
│
├── app_streamlit.py         # Streamlit UI
├── requirements.txt
└── README.md

```
## ⚙️ Setup Instructions

### 1️⃣ Create & Activate Virtual Environment
```bash
# Create Virtual Environment
  python -m venv genai_env

# Activate it:
  # On Windows:
    genai_env\Scripts\activate
  # On Mac/Linux:
    source genai_env/bin/activate
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 📄 Add Your Data
Place your documents inside:
```bash
data/sample.txt  #The chatbot answers only from this file.
```
### 3️⃣ Run the Streamlit App
```bash
streamlit run app_streamlit.py
```

### 💡 Example
Question:
```bash
Where was the Kalinga War fought?
```
Answer:
```bash
The Kalinga War was fought at Dhauli, near present-day Bhubaneswar.
```
Source:
```bash
sample.txt 
```
### ⚙️ Tech Stack

- LLM: Hugging Face (FLAN-T5)
- Embeddings: Sentence Transformers
- Vector DB: FAISS
- Framework: LangChain
- Frontend: Streamlit
- Language: Python

## 👩‍💻 Author
Anil Kumar Nayak

✨ Software Developer | Python, AI & Streamlit Enthusiast

📧 anilnayak.tech@gmail.com

### 🏁 Future Enhancements
- PDF & Web-based RAG

- Multi-document support

- OpenAI / LLaMA integration

- Chat memory

- Docker deployment
