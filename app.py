import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, render_template, request, jsonify, Response, send_file
from src.error_handler import ChatbotError, format_error_response, ConfigurationError
from src.helper import get_conversation_chain, format_response, validate_input
from embeddings import load_embeddings
import json
import logging
import os
import tempfile
from typing import Optional
from dotenv import load_dotenv

# Import voice service with error handling
try:
    from src.voice_service import get_voice_service
    VOICE_ENABLED = True
    print("✅ Voice features enabled successfully")
except Exception as e:
    print(f"⚠️ Voice service import failed: {e}")
    VOICE_ENABLED = False
    def get_voice_service():
        raise ConfigurationError("Voice service not available")

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
            print(f"\n=== QUERY PROCESSING ===")
            print(f"User query: {sanitized_input}")
            print(f"Searching vector store...")
            
            similar_docs = vector_store.similarity_search(sanitized_input, k=3)
            logger.info(f"Found {len(similar_docs)} similar documents")
            print(f"Found {len(similar_docs)} similar documents")
            
            if similar_docs:
                print(f"\n=== SIMILAR DOCUMENTS FOUND ===")
                for i, doc in enumerate(similar_docs):
                    logger.debug(f"Similar doc {i+1}: {doc.page_content[:100]}...")
                    print(f"\nDocument {i+1}:")
                    print(f"Preview: {doc.page_content[:150]}...")
                    print(f"Length: {len(doc.page_content)} characters")
                print(f"=== END SIMILAR DOCUMENTS ===")
            else:
                logger.warning("No similar documents found in vector store")
                print("No similar documents found in vector store")
            print(f"=== END QUERY PROCESSING ===\n")
                
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

@app.route('/voice-input', methods=['POST'])
def voice_input():
    """Handle voice input: record audio and convert to text."""
    if not VOICE_ENABLED:
        return jsonify({'error': 'Voice features not available'}), 503
    
    try:
        voice_service = get_voice_service()
        
        # Get timeout from request (default 20 seconds)
        timeout = request.json.get('timeout', 20) if request.is_json else 20
        
        logger.info("Processing voice input...")
        print("\n=== VOICE INPUT PROCESSING ===")
        
        # Record and transcribe audio
        audio_filepath, transcribed_text = voice_service.process_voice_input(timeout=timeout)
        
        print(f"Voice input transcribed: {transcribed_text}")
        print("=== END VOICE INPUT ===\n")
        
        return jsonify({
            'status': 'success',
            'transcribed_text': transcribed_text,
            'audio_filepath': audio_filepath
        })
        
    except ChatbotError as e:
        logger.error(f"Voice input error: {str(e)}")
        return jsonify(format_error_response(e)), 400
    except Exception as e:
        logger.error(f"Unexpected voice input error: {str(e)}")
        return jsonify(format_error_response(e)), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return audio file."""
    if not VOICE_ENABLED:
        return jsonify({'error': 'Voice features not available'}), 503
    
    try:
        voice_service = get_voice_service()
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        use_elevenlabs = data.get('use_elevenlabs', False)
        
        logger.info(f"Converting text to speech: {text[:50]}...")
        print(f"\n=== TEXT TO SPEECH ===")
        print(f"Text: {text[:100]}...")
        print(f"Using ElevenLabs: {use_elevenlabs}")
        
        # Generate audio
        audio_filepath = voice_service.generate_voice_response(
            text=text,
            use_elevenlabs=use_elevenlabs,
            autoplay=False  # Don't autoplay for web requests
        )
        
        print(f"Audio generated: {audio_filepath}")
        print("=== END TEXT TO SPEECH ===\n")
        
        # Return the audio file
        return send_file(
            audio_filepath,
            as_attachment=True,
            download_name='response.mp3',
            mimetype='audio/mpeg'
        )
        
    except ChatbotError as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        return jsonify(format_error_response(e)), 400
    except Exception as e:
        logger.error(f"Unexpected text-to-speech error: {str(e)}")
        return jsonify(format_error_response(e)), 500

@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    """Complete voice chat: voice input -> text processing -> voice output."""
    if not VOICE_ENABLED:
        return jsonify({'error': 'Voice features not available'}), 503
    
    try:
        global vector_store, chain
        voice_service = get_voice_service()
        
        # Initialize services if needed
        if vector_store is None or chain is None:
            logger.info("Services not initialized yet. Initializing lazily...")
            validate_environment()
            if vector_store is None:
                vector_store = load_embeddings()
            if chain is None:
                chain = get_conversation_chain()
            logger.info("Lazy initialization complete.")
        
        # Get parameters from request
        data = request.get_json() if request.is_json else {}
        timeout = data.get('timeout', 20)
        use_elevenlabs = data.get('use_elevenlabs', False)
        
        logger.info("Starting voice chat session...")
        print("\n=== VOICE CHAT SESSION ===")
        
        # Step 1: Record and transcribe voice input
        print("Step 1: Recording voice input...")
        audio_filepath, user_input = voice_service.process_voice_input(timeout=timeout)
        
        # Step 2: Validate input
        try:
            is_valid, sanitized_input = validate_input(user_input)
        except ChatbotError as e:
            logger.warning(f"Input validation failed: {str(e)}")
            return jsonify(format_error_response(e)), 400
        
        # Step 3: Get similar documents from vector store
        print("Step 2: Searching knowledge base...")
        similar_docs = vector_store.similarity_search(sanitized_input, k=3)
        print(f"Found {len(similar_docs)} similar documents")
        
        # Step 4: Generate text response (collect all chunks)
        print("Step 3: Generating response...")
        response_chunks = []
        for chunk in format_response(similar_docs, sanitized_input, chain):
            chunk_data = json.loads(chunk.replace('data: ', '').strip())
            if chunk_data.get('status') == 'streaming' and 'token' in chunk_data:
                response_chunks.append(chunk_data['token'])
        
        full_response = ''.join(response_chunks).strip()
        print(f"Generated response: {full_response[:100]}...")
        
        # Step 5: Convert response to speech
        print("Step 4: Converting response to speech...")
        response_audio_filepath = voice_service.generate_voice_response(
            text=full_response,
            use_elevenlabs=use_elevenlabs,
            autoplay=False
        )
        
        print("=== END VOICE CHAT SESSION ===\n")
        
        # Return both text and audio
        return jsonify({
            'status': 'success',
            'user_input': user_input,
            'text_response': full_response,
            'input_audio_filepath': audio_filepath,
            'response_audio_filepath': response_audio_filepath
        })
        
    except ChatbotError as e:
        logger.error(f"Voice chat error: {str(e)}")
        return jsonify(format_error_response(e)), 500
    except Exception as e:
        logger.error(f"Unexpected voice chat error: {str(e)}")
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
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
