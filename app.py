from flask import Flask, render_template, request, jsonify, Response
from src.error_handler import ChatbotError, format_error_response, ConfigurationError
from src.helper import get_conversation_chain, format_response, validate_input
from embeddings import load_embeddings
import json
import logging
import os
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_environment() -> None:
    """Validate required environment variables are set."""
    required_vars = ['HUGGINGFACEHUB_API_TOKEN', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ConfigurationError(
            "Missing required environment variables",
            details={"missing_vars": missing_vars}
        )

# Initialize Flask app
app = Flask(__name__)

# Placeholders for services; initialized at runtime
vector_store = None
chain = None

@app.route('/')
def home():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return streaming AI responses."""
    try:
        global vector_store, chain
        # Lazy init in case app is run under a WSGI server that skips __main__
        if vector_store is None or chain is None:
            logger.info("Services not initialized yet. Initializing lazily...")
            validate_environment()
            if vector_store is None:
                vector_store = load_embeddings()
            if chain is None:
                chain = get_conversation_chain()
            logger.info("Lazy initialization complete.")
            # Validate successful initialization
            if vector_store is None or chain is None:
                logger.error("Initialization failed: vector_store or chain is None")
                raise ChatbotError(
                    "Service initialization failed",
                    "INIT_ERROR",
                    {"vector_store_initialized": vector_store is not None, "chain_initialized": chain is not None}
                )
        # Get user input from request
        user_input = request.json.get('message', '')
        logger.info(f"Received chat request: {user_input[:50]}...")
        
        # Validate user input
        try:
            is_valid, sanitized_input = validate_input(user_input)
        except ChatbotError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            return jsonify(format_error_response(e)), 400
        
        # Get similar documents from vector store
        try:
            logger.info(f"Searching for similar documents for query: '{sanitized_input[:50]}...'")
            similar_docs = vector_store.similarity_search(sanitized_input, k=3)
            logger.info(f"Found {len(similar_docs)} similar documents")
            
            if similar_docs:
                for i, doc in enumerate(similar_docs):
                    logger.debug(f"Similar doc {i+1}: {doc.page_content[:100]}...")
            else:
                logger.warning("No similar documents found in vector store")
                
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise ChatbotError("Failed to search knowledge base", "SEARCH_ERROR", {"error": str(e)})
        
        # Set up streaming response
        def generate():
            try:
                for chunk in format_response(similar_docs, sanitized_input, chain):
                    yield chunk
            except Exception as e:
                logger.error(f"Error during response generation: {str(e)}")
                error_response = format_error_response(e)
                yield "data: " + json.dumps(error_response) + "\n\n"
        
        response = Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        return response
        
    except ChatbotError as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify(format_error_response(e)), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify(format_error_response(e)), 500

if __name__ == '__main__':
    try:
        # Validate environment
        validate_environment()
        
        # Initialize services
        logger.info("Initializing services...")
        vector_store = load_embeddings()
        chain = get_conversation_chain()
        logger.info("Services initialized successfully")
        # Validate successful initialization at startup
        if vector_store is None or chain is None:
            raise ChatbotError(
                "Service initialization failed at startup",
                "INIT_ERROR",
                {"vector_store_initialized": vector_store is not None, "chain_initialized": chain is not None}
            )
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
