from flask import Flask, render_template, request, jsonify
from src.helper import get_conversation_chain, format_response, validate_input
from embeddings import load_embeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    """Handle chat messages and return AI responses."""
    try:
        # Get user input from request
        user_input = request.json.get('message', '')
        
        # Validate user input
        is_valid, sanitized_input = validate_input(user_input)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input. Please try again.'
            }), 400
        
        # Get similar documents from vector store
        similar_docs = vector_store.similarity_search(sanitized_input, k=3)
        
        # Generate response using conversation chain
        response = format_response(similar_docs, sanitized_input, conversation)
        
        return jsonify({
            'status': 'success',
            'message': response
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure required environment variables are set
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set!")
        
    # Run the Flask app
    app.run(debug=True)
