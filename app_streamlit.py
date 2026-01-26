# import streamlit as st
# from scripts.rag_pipeline import qa_chain
# from scripts.safety import is_safe

# # Streamlit page setup
# st.set_page_config(page_title="GenAI RAG Chatbot", layout="wide")
# st.title("GenAI RAG Chatbot")
# st.write("Ask me anything based on the documents you have!")

# # User input
# question = st.text_input("Ask me anything:")

# if question:
#     if not is_safe(question):
#         st.error("Unsafe content detected!")
#     else:
#         with st.spinner("Thinking..."):
#             # Call the RAG chain
#             result1 = qa_chain(question)  # returns dict
#             answer = result1['result']
#             sources = result1['source_documents']

#         # Show concise answer
#         st.subheader("Answer")
#         st.success(answer)  # green box for answer

#         # Show source documents in collapsible sections
#         st.subheader("Source Documents")
#         for i, doc in enumerate(sources, 1):
#             with st.expander(f"Document {i}"):
#                 st.write(doc.page_content)





#=========================================================================


import logging
import streamlit as st
from scripts.rag_pipeline import qa_chain
from scripts.safety import is_safe

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Streamlit application started")

# --------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------
st.set_page_config(page_title="GenAI RAG Chatbot", layout="wide")
st.title("🤖 GenAI RAG Chatbot")
st.write("Python version:", sys.version)
st.write("Ask questions based on the uploaded documents.")

# --------------------------------------------------
# User Input
# --------------------------------------------------
question = st.text_input("Ask me anything:")

if question:
    logger.info(f"User question received: {question}")

    if not is_safe(question):
        logger.warning("Unsafe user input detected")
        st.error("🚫 Unsafe content detected!")
    else:
        try:
            with st.spinner("Thinking..."):
                logger.info("Sending question to RAG pipeline")
                result = qa_chain(question)

            answer = result["result"]
            sources = result["source_documents"]

            logger.info("Answer generated successfully")

            # Display Answer
            st.subheader("Answer")
            st.success(answer)

            # Display Sources
            st.subheader("Source Documents")
            for i, doc in enumerate(sources, 1):
                with st.expander(f"Document {i}"):
                    st.write(doc.page_content)

        except Exception as e:
            logger.error(f"Error while processing question: {e}", exc_info=True)
            st.error("❌ Something went wrong while generating the answer.")
