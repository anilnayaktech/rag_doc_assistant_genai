# from langchain.schema import Document
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# class EmbeddingStore:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)
#         self.index = None
#         self.texts = []
#         self.faiss_store = None

#     def add_texts(self, texts):
#         embeddings = self.model.encode(texts)
#         self.texts.extend(texts)

#         # Create or update FAISS index
#         if self.index is None:
#             dim = embeddings.shape[1]
#             self.index = faiss.IndexFlatL2(dim)
#         self.index.add(np.array(embeddings, dtype='float32'))

#         # Wrap FAISS with LangChain retriever
#         docs = [Document(page_content=t) for t in self.texts]
#         hf_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
#         self.faiss_store = FAISS.from_documents(docs, hf_embeddings)


    
#     # ✅ Add 'k' parameter here for top-k retrieval
#     def as_retriever(self, k):
#         if self.faiss_store is None:
#             raise ValueError("No documents added yet")
#         return self.faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": k})




#=============================================================================



import logging
import faiss
import numpy as np
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model and internal FAISS structures.
        """
        logger.info(f"Initializing EmbeddingStore with model: {model_name}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.faiss_store = None

        logger.info("Embedding model loaded successfully")

    def add_texts(self, texts: list):
        """
        Converts input texts into embeddings and stores them in FAISS.
        """
        if not texts:
            logger.warning("No texts provided to add_texts()")
            return

        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts)

        self.texts.extend(texts)

        # Create FAISS index if not exists
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            logger.info(f"Created new FAISS index with dimension: {dim}")

        self.index.add(np.array(embeddings, dtype="float32"))
        logger.info(f"Added {len(texts)} embeddings to FAISS index")

        # Wrap FAISS with LangChain retriever
        docs = [Document(page_content=t) for t in self.texts]
        hf_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.faiss_store = FAISS.from_documents(docs, hf_embeddings)

        logger.info("FAISS vector store wrapped successfully with LangChain")

    def as_retriever(self, k: int):
        """
        Returns a LangChain retriever for top-k similarity search.
        """
        if self.faiss_store is None:
            logger.error("Retriever requested before adding documents")
            raise ValueError("No documents added yet. Call add_texts() first.")

        logger.info(f"Creating retriever with top-k = {k}")
        return self.faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
