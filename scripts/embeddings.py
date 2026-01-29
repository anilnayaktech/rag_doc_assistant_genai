import logging
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Note: We removed 'import faiss' and 'import numpy' because 
# LangChain's FAISS wrapper handles them for us.

logger = logging.getLogger(__name__)

# This function ensures the 400MB embedding model is loaded ONLY ONCE
@st.cache_resource
def get_embedding_model(model_name):
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing EmbeddingStore with model: {model_name}")
        self.model_name = model_name
        self.faiss_store = None
        # We NO LONGER load SentenceTransformer here to save 50% RAM
        
    def add_texts(self, texts: list):
        if not texts:
            logger.warning("No texts provided to add_texts()")
            return

        logger.info(f"Generating embeddings for {len(texts)} texts using LangChain wrapper")
        
        # This single line handles the model loading AND the vectorization
        #hf_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        # Use the cached model instead of creating a new one
        hf_embeddings = get_embedding_model(self.model_name)
        
        docs = [Document(page_content=t) for t in texts]
        
        # This line creates the FAISS index automatically
        self.faiss_store = FAISS.from_documents(docs, hf_embeddings)

        logger.info("FAISS vector store created successfully")

    def as_retriever(self, k: int):
        if self.faiss_store is None:
            raise ValueError("No documents added yet. Call add_texts() first.")

        return self.faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
