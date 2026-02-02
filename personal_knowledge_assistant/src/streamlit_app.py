"""
Streamlit Web UI - Upload documents and chat with them
"""

import streamlit as st
from pathlib import Path

from src.document_processor import process_documents
from src.rag_chain import answer_question
from langchain_community.vectorstores import Chroma
from src.embeddings_manager import embeddings
from src.retriever import CHROMA_PATH
from src.utils import logger


st.set_page_config(page_title="Knowledge Base Assistant", layout="wide")

st.title("📚 Personal Knowledge Base Assistant")

st.markdown("Upload documents and ask questions about them!")


# Sidebar for document upload
with st.sidebar:
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if st.button("Upload"):
        if uploaded_files:
            # Create temp directory
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            # Save uploaded files
            for file in uploaded_files:
                with open(temp_dir / file.name, "wb") as f:
                    f.write(file.getbuffer())
            
            # Process documents
            st.info("Processing documents...")
            chunks = process_documents(temp_dir)
            
            if chunks:
                # Store in Chroma
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(CHROMA_PATH)
                )
                st.success(f"✓ Stored {len(chunks)} chunks from {len(uploaded_files)} files")
                logger.info(f"Uploaded {len(uploaded_files)} files")
            else:
                st.error("❌ Failed to process documents - Files may be empty or corrupted. Try uploading different files.")
        else:
            st.warning("Please select files first")


# Main chat interface
st.header("💬 Ask Questions")

question = st.text_input("Ask a question about your documents:")

if question:
    if not CHROMA_PATH.exists():
        st.error("No documents in vector store. Upload documents first!")
    else:
        st.info(f"📝 Question: {question}")
        
        try:
            with st.spinner("🔍 Searching documents and generating answer..."):
                answer = answer_question(question)
            
            if answer:
                st.success("✓ Answer Generated!")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.error("Failed to generate answer. Check logs for details.")
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            logger.error(f"Error in answer_question: {e}")


# Footer
st.markdown("---")
st.markdown("Built with LangChain + Streamlit")
