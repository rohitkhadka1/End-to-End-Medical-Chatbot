# Mental Health Chatbot

A comprehensive mental health support chatbot built with Flask, LangChain, and Hugging Face models. The chatbot provides empathetic responses based on a knowledge base derived from mental health care manuals.

## Features

- 🤖 AI-powered mental health support using Hugging Face models
- 📚 Knowledge base from mental health care manuals
- 🔍 Semantic search using Pinecone vector database
- 💬 Real-time streaming responses
- 🎨 Modern, responsive web interface
- 🛡️ Comprehensive error handling and validation

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Hugging Face account and API token
- Pinecone account and API key

### 2. Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd End-to-End-Medical-Chatbot
```

2. Create a virtual environment:
```bash
python -m venv chatbot
# On Windows:
chatbot\Scripts\activate
# On macOS/Linux:
source chatbot/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirement.txt
```

### 3. Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=mental-health-chatbot
```

### 4. Get API Keys

#### Hugging Face API Token:
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Copy the token to your `.env` file

#### Pinecone API Key:
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new project or use existing one
3. Copy the API key and environment to your `.env` file

### 5. Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
End-to-End-Medical-Chatbot/
├── app.py                 # Main Flask application
├── embeddings.py          # Vector store and embeddings management
├── requirement.txt        # Python dependencies
├── .env.example          # Environment variables template
├── src/
│   ├── helper.py         # LLM chain and response formatting
│   ├── error_handler.py  # Error handling utilities
│   └── a-manual-of-mental-health-care-in-general-practice.pdf
├── templates/
│   └── chat.html         # Web interface
└── README.md

```

## Key Components

### 1. LLM Chain (`src/helper.py`)
- Uses Google's FLAN-T5 model for reliable text generation
- Custom prompt template for mental health conversations
- Streaming response simulation for better UX

### 2. Vector Store (`embeddings.py`)
- Sentence Transformers for document embeddings
- Pinecone for scalable vector storage
- Automatic PDF processing and chunking

### 3. Web Interface (`templates/chat.html`)
- Modern, responsive design
- Real-time streaming responses
- Error handling and user feedback

### 4. Error Handling (`src/error_handler.py`)
- Comprehensive error types
- Retry mechanisms with exponential backoff
- Detailed logging and user-friendly messages

## Usage

1. Start a conversation by typing a mental health related question
2. The chatbot will search its knowledge base for relevant information
3. Responses are generated using the LLM with context from the knowledge base
4. All conversations include appropriate disclaimers about professional help

## Troubleshooting

### Common Issues

1. **"HUGGINGFACEHUB_API_TOKEN not set"**
   - Ensure your `.env` file exists and contains the correct token
   - Verify the token has proper permissions

2. **"PINECONE_API_KEY not set"**
   - Check your Pinecone API key and environment settings
   - Ensure the Pinecone index is created (done automatically)

3. **"PDF file not found"**
   - Ensure the PDF file exists in the `src/` directory
   - Check file permissions

4. **Model initialization errors**
   - Check internet connection for model downloads
   - Verify Hugging Face API token validity

### Logs

The application provides detailed logging. Check the console output for:
- Service initialization status
- Error messages with details
- Response generation progress

## Development

### Adding New Features

1. **New Models**: Update `src/helper.py` to use different Hugging Face models
2. **Custom Knowledge Base**: Replace the PDF in `src/` directory
3. **UI Improvements**: Modify `templates/chat.html`
4. **API Extensions**: Add new routes in `app.py`

### Testing

Run the application in debug mode:
```bash
python app.py
```

The Flask debug mode provides detailed error information and auto-reloading.

## Security Considerations

- API keys are stored in environment variables
- Input validation prevents malicious inputs
- Error messages don't expose sensitive information
- Rate limiting can be added for production use

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Create an issue on GitHub with detailed information