
import logging
import torch
import streamlit as st

# NEW (For 2026/Current LangChain)
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from scripts.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Cached Resource Loader
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading GenAI Models... Please wait.")
def initialize_rag_pipeline():
    """
    Loads models and data once and keeps them in memory.
    """
    logger.info("Initializing RAG pipeline (Cached Run)")
    
    # 1. Load LLM with float16 to save 50% RAM (Critical for Cloud)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We use torch.float16 to stay under the 1GB RAM limit
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    text_gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.2
    )
    llm = HuggingFacePipeline(pipeline=text_gen)

    # 2. Setup Embeddings and Documents
    store = EmbeddingStore()
    try:
        with open("data/sample.txt", encoding="utf-8") as f:
             # docs = [f.read()]
            full_text = f.read()
         # Split the text into manageable chunks for the 512-token limit
        from langchain_text_splitters import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(full_text)
          
        store.add_texts(docs)
    except FileNotFoundError:
        logger.error("data/sample.txt not found! Please check your file structure.")
        # Create a fallback so the app doesn't crash
        store.add_texts(["Welcome to the chatbot. No data found."])

    retriever = store.as_retriever(k=3)

    # 3. Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. "
            "Answer the question in 1–2 concise sentences using the context below. "
            "If the answer is not in the context, say: 'I don't know.'\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    # 4. Create Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

# This replaces the global code to prevent double-loading
qa_chain = initialize_rag_pipeline()

