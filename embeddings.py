from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import os
import logging
import time
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SentenceTransformerEmbeddings:
    """Custom embeddings wrapper for SentenceTransformer to work with LangChain"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model
        
        Args:
            model_name: HuggingFace model name for sentence transformers
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Extract text content if documents are LangChain Document objects
            text_content = []
            for text in texts:
                if hasattr(text, 'page_content'):
                    text_content.append(text.page_content)
                else:
                    text_content.append(str(text))
            
            embeddings = self.model.encode(
                text_content, 
                show_progress_bar=True, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(
                [text], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            raise

def _require_pinecone_env() -> None:
    """Ensure Pinecone API key exists in environment for langchain-pinecone to use."""
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env file")

def _ensure_index_exists(index_name: str, dimension: int) -> None:
    """Create the Pinecone index if it does not already exist (v3 client)."""
    api_key = os.getenv('PINECONE_API_KEY')
    region = os.getenv('PINECONE_REGION', 'us-east-1')
    pc = PineconeClient(api_key=api_key)
    
    # Get list of indexes - handle different API response formats
    try:
        indexes_response = pc.list_indexes()
        # Handle both old and new API formats
        if hasattr(indexes_response, 'indexes'):
            # New API format
            existing = [idx.name for idx in indexes_response.indexes]
        elif isinstance(indexes_response, list):
            # Handle list of index objects
            existing = [idx.name if hasattr(idx, 'name') else str(idx) for idx in indexes_response]
        else:
            # Handle direct list of strings
            existing = [str(idx) for idx in indexes_response]
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        existing = []
    
    logger.info(f"Checking if index '{index_name}' exists...")
    logger.info(f"Existing indexes: {existing}")
    
    if index_name not in existing:
        logger.info(f"Creating new Pinecone index '{index_name}' with dimension {dimension}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        logger.info(f"Created Pinecone index '{index_name}' in region '{region}'")
        
        # wait for ready
        max_wait_time = 60  # Maximum wait time in seconds
        wait_time = 0
        while wait_time < max_wait_time:
            try:
                status = pc.describe_index(index_name).status
                if status.get('ready', False):
                    logger.info(f"Index '{index_name}' is ready!")
                    break
                else:
                    logger.info(f"Waiting for index to be ready... ({wait_time}s)")
                    time.sleep(5)
                    wait_time += 5
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
                time.sleep(5)
                wait_time += 5
        
        if wait_time >= max_wait_time:
            logger.warning(f"Index creation timeout after {max_wait_time}s")
    else:
        logger.info(f"Index '{index_name}' already exists")

def create_embeddings(index_name: str = "mental-health-chatbot") -> PineconeVectorStore:
    """
    Create embeddings from the mental health manual PDF and store them in Pinecone.
    
    Args:
        index_name (str): Name of the Pinecone index to create
        
    Returns:
        PineconeVectorStore: The vector store for similarity search
    """
    try:
        # Ensure Pinecone credentials are available; langchain-pinecone will handle client and index creation
        _require_pinecone_env()

        # Check if PDF file exists
        pdf_path = os.path.join('src', 'a-manual-of-mental-health-care-in-general-practice.pdf')
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        # Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents were loaded from the PDF")
        
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Split documents into {len(texts)} chunks")

        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

        # Ensure index exists before upserting
        try:
            _ensure_index_exists(index_name=index_name, dimension=384)
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {str(e)}")
            raise

        # Create the vector store in Pinecone by upserting all chunks.
        try:
            logger.info(f"Creating vector store with {len(texts)} document chunks...")
            vector_store = PineconeVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                index_name=index_name
            )
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
        
        logger.info(f"Successfully created embeddings in Pinecone index: {index_name}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {str(e)}")
        raise

def load_embeddings(index_name: str = "mental-health-chatbot") -> Optional[PineconeVectorStore]:
    """
    Load the embeddings from Pinecone.
    
    Args:
        index_name (str): Name of the Pinecone index to load
        
    Returns:
        Optional[PineconeVectorStore]: The vector store for similarity search
    """
    try:
        _require_pinecone_env()

        # Create embeddings model
        embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

        try:
            # Ensure index exists before loading
            _ensure_index_exists(index_name=index_name, dimension=384)
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            logger.info(f"Successfully loaded embeddings from Pinecone index: {index_name}")
            return vector_store
        except Exception as load_err:
            logger.warning(f"Could not load existing Pinecone index '{index_name}': {load_err}. Creating embeddings...")
            return create_embeddings(index_name)

    except Exception as e:
        logger.error(f"Failed to load or create embeddings: {str(e)}")
        raise

def get_similar_docs(query: str, vector_store: PineconeVectorStore, k: int = 3) -> list:
    """
    Search for similar documents in the vector store based on the query.
    
    Args:
        query (str): The search query
        vector_store (PineconeVectorStore): The vector store to search in
        k (int): Number of similar documents to retrieve
        
    Returns:
        list: List of similar documents
    """
    try:
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        similar_docs = vector_store.similarity_search(query, k=k)
        logger.debug(f"Found {len(similar_docs)} similar documents for query: {query[:50]}...")
        return similar_docs
    except Exception as e:
        logger.error(f"Error searching for similar documents: {str(e)}")
        raise

print("Embeddings module loaded. Call load_embeddings() to initialize the vector store.")