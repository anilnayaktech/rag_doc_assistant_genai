
import logging
import streamlit as st
import sys
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
# st.write("Python version:", sys.version)
st.write("Ask questions based on the uploaded documents.")

# --------------------------------------------------
# User Input
# --------------------------------------------------

#question = st.text_input("Ask me anything:")
with st.form(key="chat_form", clear_on_submit=False):
    # The text input inside the form
    question = st.text_input("Ask me anything:", placeholder="e.g., Who built the Konark Sun Temple?")
    
    # The dedicated button
    submit_button = st.form_submit_button(label="Ask Question", type="primary")

# Logic triggers only when the button is clicked
if submit_button and question:
#if question:
    logger.info(f"User question received: {question}")

    if not is_safe(question):
        logger.warning("Unsafe user input detected")
        st.error("🚫 Unsafe content detected!")
    else:
        try:
            with st.spinner("Thinking..."):
                logger.info("Sending question to RAG pipeline")
                # Ensure qa_chain is using .invoke() for 2026 LangChain standards
                result = qa_chain.invoke(question)
                #result = qa_chain(question)


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
