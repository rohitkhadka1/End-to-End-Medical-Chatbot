from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from typing import Optional, List
import os
import logging
import time
import numpy as np
from dotenv import load_dotenv

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

def init_pinecone() -> PineconeClient:
    """
    Initialize Pinecone client with API key from .env file.
    Returns:
        PineconeClient: Initialized Pinecone client
    """
    try:
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set in .env file")
            
        pc = PineconeClient(api_key=api_key)
        logger.info("Pinecone initialized successfully")
        return pc
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

def create_embeddings(index_name: str = "mental-health-chatbot") -> PineconeVectorStore:
    """
    Create embeddings from the mental health manual PDF and store them in Pinecone.
    
    Args:
        index_name (str): Name of the Pinecone index to create
        
    Returns:
        PineconeVectorStore: The vector store for similarity search
    """
    try:
        # Initialize Pinecone
        pc = init_pinecone()
        
        # Create Pinecone index if it doesn't exist
        dimension = 384  # dimension for 'all-MiniLM-L6-v2' model
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Created new Pinecone index: {index_name}")
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)
        
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
        
        # Create the vector store in Pinecone (v3 SDK compatible)
        vector_store = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=index_name
        )
        
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
        # Initialize Pinecone
        pc = init_pinecone()
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.warning(f"Pinecone index {index_name} does not exist. Creating new embeddings...")
            return create_embeddings(index_name)
            
        # Create embeddings model
        embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        
        # Load the existing index (v3 SDK compatible)
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        logger.info(f"Successfully loaded embeddings from Pinecone index: {index_name}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to load embeddings: {str(e)}")
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