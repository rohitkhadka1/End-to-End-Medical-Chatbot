from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_embeddings():
    """
    Create embeddings from the mental health manual PDF and save them to a vector store.
    Returns the vector store for similarity search.
    """
    # Load the PDF document
    pdf_path = os.path.join('src', 'a-manual-of-mental-health-care-in-general-practice.pdf')
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings using a HuggingFace model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save the vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Save the vector store locally
    vector_store.save_local("faiss_index")
    
    return vector_store

def load_embeddings():
    """
    Load the saved embeddings from the vector store.
    Returns the vector store for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the vector store if it exists, otherwise create new embeddings
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings)
    else:
        vector_store = create_embeddings()
    
    return vector_store

def get_similar_docs(query, vector_store, k=3):
    """
    Search for similar documents in the vector store based on the query.
    
    Args:
        query (str): The search query
        vector_store: The FAISS vector store
        k (int): Number of similar documents to retrieve
        
    Returns:
        list: List of similar documents
    """
    similar_docs = vector_store.similarity_search(query, k=k)
    return similar_docs
