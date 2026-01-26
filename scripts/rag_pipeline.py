# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from scripts.embeddings import EmbeddingStore
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# # Load Hugging Face LLM
# # -------------------------
# # Use an instruction-tuned model instead of GPT-2
# # (much better at following "Answer concisely" instructions)
# model_name = "google/flan-t5-base"   # You can also try flan-t5-large  
# # model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Use text2text-generation (not text-generation!)
# text_gen = pipeline(
#     "text2text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=50,   # Keep answers short
#     temperature=0.2        # Deterministic output
# )

# llm = HuggingFacePipeline(pipeline=text_gen)

# # -------------------------
# # Load embeddings & retriever
# # -------------------------
# store = EmbeddingStore()

# # Read entire file as a single document
# # docs = open("data/sample.txt").read().split("\n")
# docs = [open("data/sample.txt").read()]
# store.add_texts(docs)


# # Retrieve only top-1 most relevant doc
# retriever = store.as_retriever(k=3)


# # -------------------------
# # Concise prompt template
# # -------------------------
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         # "You are a helpful assistant. "
#         # "Answer the following question in ONE short sentence, based only on the given context. "
#         # "If the answer is not in the context, reply: 'I don't know.'\n\n"
#         # "Context: {context}\n\n"
#         # "Question: {question}\n\n"
#         # "Answer:"

#         #==================================
#         # "You are a helpful assistant. "
#         # "Answer the following question in a **concise sentence (1-2 sentences)**, "
#         # "using the context below. Include necessary details, like years, events, or names. "
#         # "If the answer is not in the context, reply: 'I don't know.'\n\n"
#         # "Context: {context}\n\n"
#         # "Question: {question}\n\n"
#         # "Answer:"

#         #=====================================
#         "You are a helpful assistant. "
#         "Answer the following question **in 1-2 concise sentences**, "
#         "using the context below. Make the answer **naturally readable** like a human-written sentence, "
#         "even if it means slightly rephrasing the context. "
#         "If the answer is not in the context, reply: 'I don't know.'\n\n"
#         "Context: {context}\n\n"
#         "Question: {question}\n\n"
#         "Answer:"
# # =========================================
#         #  "You are a helpful assistant. "
#         # "Answer the following question in 1-2 concise sentences, using the context below. "
#         # "Whenever possible, combine names, dynasties, years, and relevant details from the context into a clear, complete answer. "
#         # "If the answer is not in the context, reply 'I don't know.'\n\n"
#         # "Context: {context}\n\n"
#         # "Question: {question}\n\n"
#         # "Answer:"
#     )
# )

# # -------------------------
# # RetrievalQA chain
# # -------------------------
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt_template},
#     return_source_documents=True
# )
# # -------------------------
# # Debug/test mode (runs ONLY if this file is executed directly)
# # -------------------------
# if __name__ == "__main__":
#     query = "What is the capital city of Odisha?"
#     result = qa_chain(query)

#     print("Answer:", result['result'])
#     print("\nSources:")
#     for i, doc in enumerate(result['source_documents']):
#         print(f"Doc {i+1}: {doc.page_content[:200]}...")




#===================================================================================



import logging
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from scripts.embeddings import EmbeddingStore

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting RAG pipeline initialization")

# --------------------------------------------------
# Load Hugging Face LLM
# --------------------------------------------------
model_name = "google/flan-t5-base"
logger.info(f"Loading LLM model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text_gen = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=80,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=text_gen)
logger.info("LLM loaded successfully")

# --------------------------------------------------
# Load Embeddings & Documents
# --------------------------------------------------
store = EmbeddingStore()

logger.info("Loading documents from data/sample.txt")
docs = [open("data/sample.txt", encoding="utf-8").read()]
store.add_texts(docs)

logger.info("Documents embedded successfully")

# --------------------------------------------------
# Retriever Configuration
# --------------------------------------------------
TOP_K = 3
retriever = store.as_retriever(k=TOP_K)
logger.info(f"Retriever created with top-k = {TOP_K}")

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. "
        "Answer the following question in 1–2 concise sentences using the context below. "
        "Make the answer natural and human-readable. "
        "If the answer is not in the context, reply: 'I don't know.'\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

logger.info("Prompt template initialized")

# --------------------------------------------------
# RetrievalQA Chain
# --------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

logger.info("RAG QA chain created successfully")

# --------------------------------------------------
# Debug / Test Mode
# --------------------------------------------------
if __name__ == "__main__":
    logger.info("Running RAG pipeline in debug mode")

    query = "What is the capital city of Odisha?"
    result = qa_chain(query)

    logger.info("Query processed successfully")
    print("Answer:", result["result"])
