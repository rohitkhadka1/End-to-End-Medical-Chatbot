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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_environment() -> None:
    """Validate required environment variables are set."""
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ConfigurationError(
            "Missing required environment variables",
            details={"missing_vars": missing_vars}
        )

# Initialize Flask app
app = Flask(__name__)

# Load vector store and conversation chain
vector_store = load_embeddings()
conversation = get_conversation_chain()

@app.route('/')
def home():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return streaming AI responses."""
    try:
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
            similar_docs = vector_store.similarity_search(sanitized_input, k=3)
            logger.debug(f"Found {len(similar_docs)} similar documents")
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise ChatbotError("Failed to search knowledge base", "SEARCH_ERROR", {"error": str(e)})
        
        # Set up streaming response
        def generate():
            try:
                for chunk in format_response(similar_docs, sanitized_input, conversation):
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
        conversation = get_conversation_chain()
        logger.info("Services initialized successfully")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
