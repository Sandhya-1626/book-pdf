import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Aero-GPT | Propulsion Research Portal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #238636;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    .chat-container {
        padding: 20px;
        border-radius: 12px;
        background-color: #161b22;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    .status-ready {
        color: #3fb950;
        font-weight: bold;
    }
    .status-wait {
        color: #f85149;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 Aero-GPT")
    st.markdown("### Research Portal")
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Upload Propulsion Research PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Status Indicator
    if st.session_state.vector_store:
        st.markdown('Status: <span class="status-ready">Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('Status: <span class="status-wait">Waiting for Data...</span>', unsafe_allow_html=True)

# --- CORE LOGIC (OPTIMIZED) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs(files):
    all_docs = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"Processing: {file.name}...")
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # PyMuPDFLoader is ~10x faster than PyPDFLoader
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            all_docs.extend(docs)
        finally:
            os.remove(tmp_path)
        
        progress_bar.progress((i + 1) / len(files))

    status_text.text("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    
    status_text.text("Generating vector embeddings (First time may take a minute)...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    status_text.empty()
    progress_bar.empty()
    return vector_store

if uploaded_files and not st.session_state.vector_store:
    st.session_state.vector_store = process_pdfs(uploaded_files)
    st.rerun()

# --- CHAT INTERFACE ---
st.title("Antigravity & Theoretical Propulsion Research Assistant")
st.markdown("Analyze complex physics papers with AI-powered RAG.")

# Display Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Ask a question about the research papers..."):
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Aero-GPT is thinking..."):
                try:
                    # Setup LLM (Ollama)
                    llm = ChatOllama(model="llama3", temperature=0.2)
                    
                    # Alternative OpenAI (Commented out as requested)
                    # from langchain_openai import ChatOpenAI
                    # llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key="YOUR_KEY")

                    # System Prompt logic
                    system_prompt = (
                        "You are Aero-GPT, a Physics Research Assistant. You specialize in Antigravity, General Relativity, and Propulsion. "
                        "Use ONLY the following pieces of retrieved context to answer the question. "
                        "If the answer isn't in the context, say you don't know based on the papers. "
                        "Use LaTeX ($...$) for all math/physics equations. "
                        "Maintain a highly professional, academic tone.\n\n"
                        "{context}"
                    )
                    
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), question_answer_chain)
                    
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    # Extract Sources
                    sources = []
                    for doc in response["context"]:
                        source_info = f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
                        if source_info not in sources:
                            sources.append(source_info)
                    
                    full_response = answer
                    if sources:
                        full_response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                    
                    st.markdown(full_response)
                    st.session_state.chat_history.append(AIMessage(content=full_response))
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if "ollama" in str(e).lower():
                        st.warning("Make sure Ollama is running and Llama3 is pulled.")
    else:
        st.warning("Please upload PDFs first to initialize the knowledge base.")

# --- FOOTER ---
st.divider()
st.caption("Developed by Antigravity | AI Physics Researcher Portal")
