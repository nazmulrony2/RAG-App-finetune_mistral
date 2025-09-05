# app.py
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
import tempfile
import os
import time
from typing import List

# --- Configuration ---
CHROMA_DB_PATH = "vector_db_ollama_nomic"
OLLAMA_MODEL = "mistral:latest"  # Change this if you use a different model
EMBEDDING_MODEL = "nomic-embed-text"  # Your chosen embedding model

# --- Helper Functions ---
@st.cache_resource
def load_embedding_model():
    """Load the Ollama embedding model"""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        st.sidebar.success(f"✅ Embedding model '{EMBEDDING_MODEL}' loaded")
        return embeddings
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_vectorstore(_embeddings):
    """Load or create the vector database"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=_embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def process_uploaded_files(uploaded_files, _embeddings):
    """Process uploaded files and add to vector database"""
    if not uploaded_files:
        return None
    
    all_docs = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(tmp_file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(tmp_file_path)
            else:
                st.warning(f"Skipped unsupported file: {uploaded_file.name}")
                continue
                
            documents = loader.load()
            all_docs.extend(documents)
            st.sidebar.info(f"📄 Loaded {uploaded_file.name} ({len(documents)} pages)")
            
        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.unlink(tmp_file_path)
    
    if not all_docs:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    
    chunks = text_splitter.split_documents(all_docs)
    st.sidebar.info(f"✂️ Split into {len(chunks)} chunks")
    
    # Create vector store
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        st.sidebar.success(f"✅ Added {len(chunks)} chunks to vector database")
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {e}")
        return None

def get_relevant_context(query, vectorstore, k=3):
    """Retrieve relevant context from vector database"""
    try:
        results = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
                              for doc in results])
        return context, len(results)
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "Error retrieving context.", 0

def generate_response(query, context, conversation_history=[]):
    """Generate response using Ollama with context and conversation history"""
    
    # Build conversation history prompt
    history_prompt = ""
    if conversation_history:
        history_prompt = "\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_prompt += f"{role}: {msg['content']}\n"
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information to answer the question clearly, say "I don't have enough information to answer that based on the provided documents."

Context from documents:
{context}
{history_prompt}

Current question: {query}

Instructions:
- Be concise and factual
- Base your answer strictly on the context provided
- If unsure, say you don't know rather than guessing
- Format your response clearly

Answer:"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_ctx': 4096  # Context window size
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"❌ Error generating response: {str(e)}. Please ensure Ollama is running."

def check_ollama_status():
    """Check if Ollama is running and models are available"""
    try:
        # Check if Ollama is running
        models = ollama.list()
        model_names = [model['name'] for model in models.get('models', [])]
        
        status = {
            'ollama_running': True,
            'mistral_available': OLLAMA_MODEL in model_names,
            'embedding_available': any(EMBEDDING_MODEL in name for name in model_names),
            'models': model_names
        }
        return status
    except:
        return {'ollama_running': False, 'mistral_available': False, 'embedding_available': False, 'models': []}

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Chat with your Docs - Ollama RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("💬 ChatGPT-like Chat with Your Documents")
    st.caption(f"Powered by Ollama ({OLLAMA_MODEL}) + {EMBEDDING_MODEL} embeddings + RAG")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check Ollama status
        ollama_status = check_ollama_status()
        
        if ollama_status['ollama_running']:
            st.success("✅ Ollama is running")
            if ollama_status['mistral_available']:
                st.success(f"✅ {OLLAMA_MODEL} model available")
            else:
                st.error(f"❌ {OLLAMA_MODEL} not found. Available: {', '.join(ollama_status['models'])}")
            
            if ollama_status['embedding_available']:
                st.success(f"✅ {EMBEDDING_MODEL} embedding model available")
            else:
                st.warning(f"⚠️ {EMBEDDING_MODEL} not found. Available: {', '.join(ollama_status['models'])}")
        else:
            st.error("❌ Ollama is not running")
            st.info("Please start Ollama with: `ollama serve`")
        
        st.divider()
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'md', 'docx', 'doc'],
            accept_multiple_files=True
        )
        
        chunks_to_retrieve = st.slider("Number of context chunks", 1, 5, 3)
        
        if uploaded_files and st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                embeddings = load_embedding_model()
                if embeddings:
                    new_vectorstore = process_uploaded_files(uploaded_files, embeddings)
                    if new_vectorstore:
                        st.session_state.vectorstore = new_vectorstore
                        st.session_state.documents_processed = True
                        st.rerun()
        
        if st.session_state.vectorstore:
            st.success(f"📊 Vector database: {st.session_state.vectorstore._collection.count()} chunks")
        
        st.divider()
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.documents_processed = False
            if os.path.exists(CHROMA_DB_PATH):
                import shutil
                shutil.rmtree(CHROMA_DB_PATH)
            st.rerun()

    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show context sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View sources"):
                        st.info(message["sources"])

    with col2:
        st.header("ℹ️ Info")
        if not st.session_state.documents_processed:
            st.info("""
            **To get started:**
            1. Ensure Ollama is running
            2. Upload documents in the sidebar
            3. Click 'Process Documents'
            4. Start chatting!
            """)
        else:
            st.success("✅ Documents ready for querying")
            st.metric("Chunks in database", st.session_state.vectorstore._collection.count())

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            if st.session_state.vectorstore:
                # Retrieve relevant context
                context, num_sources = get_relevant_context(
                    prompt, 
                    st.session_state.vectorstore, 
                    k=chunks_to_retrieve
                )
                
                # Generate response
                full_response = generate_response(
                    prompt, 
                    context,
                    st.session_state.messages
                )
                
                # Display response
                message_placeholder.markdown(full_response)
                
                # Store message with sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": f"Used {num_sources} context chunks:\n\n{context}"
                })
            else:
                error_msg = "No documents processed. Please upload and process documents first."
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

if __name__ == "__main__":
    main()