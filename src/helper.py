from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import logging
from typing import Generator, Optional
from .error_handler import (ChatbotError, APIError, ValidationError, ConfigurationError,
    retry_with_exponential_backoff, format_error_response
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry_with_exponential_backoff(max_attempts=3, max_wait=10)
def get_conversation_chain() -> ConversationChain:
    """
    Create a conversation chain with memory and custom prompt template.
    Returns the conversation chain object.
    
    Raises:
        ConfigurationError: If API key is missing or invalid
        APIError: If there's an error initializing the language model
    """
    try:
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY environment variable is not set",
                details={"required_env_var": "OPENAI_API_KEY"}
            )
            
        # Initialize language model with OpenAI
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key,
            request_timeout=30,  # Timeout for API requests in seconds
            max_retries=3,  # Number of retries for failed requests
            streaming=True,  # Enable streaming responses
            callbacks=[],  # Will be set per-request for streaming
        )
        
        # Test the LLM connection
        try:
            llm.predict("test")
        except Exception as e:
            raise APIError(
                "Failed to initialize language model",
                details={"error": str(e)}
            )
        
        logger.info("Successfully initialized language model")
        
        # Create a custom prompt template for mental health conversations
        template = """You are a helpful and empathetic mental health assistant, trained to provide supportive and professional guidance.
        Use the provided context to offer evidence-based support and information while maintaining appropriate boundaries.
        Always maintain a professional, compassionate, and supportive tone. If you're unsure about something, acknowledge it and suggest consulting a mental health professional.

        Important Guidelines:
        - Focus on providing general support and information
        - Do not attempt to diagnose conditions
        - Encourage professional help when appropriate
        - Maintain confidentiality and empathy
        - Use inclusive and respectful language

        Context from mental health manual: {context}

        Current conversation:
        {history}
        Human: {input}
        Assistant:"""
        
        # Create the prompt template with input validation
        try:
            prompt = PromptTemplate(
                input_variables=["context", "history", "input"],
                template=template,
                validate_template=True  # Validates that all input variables are present in the template
            )
        except Exception as e:
            raise ConfigurationError(
                "Error creating prompt template",
                details={"error": str(e)}
            )
        
        # Set up conversation memory with error handling
        try:
            memory = ConversationBufferMemory(
                return_messages=True,
                human_prefix="Human",
                ai_prefix="Assistant",
                input_key="input",  # Specify the input key to match the prompt template
                output_key="output"  # Specify the output key for consistency
            )
        except Exception as e:
            raise ConfigurationError(
                "Error setting up conversation memory",
                details={"error": str(e)}
            )
        
        # Create the conversation chain with comprehensive configuration
        try:
            conversation = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=prompt,
                verbose=True,
                return_final_only=True,  # Only return the final response
                max_iterations=1  # Prevent infinite loops
            )
            return conversation
        except Exception as e:
            raise ConfigurationError(
                "Error creating conversation chain",
                details={"error": str(e)}
            )
            
    except (ConfigurationError, APIError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error during conversation chain initialization: {str(e)}")
        raise ConfigurationError(
            "Failed to initialize conversation chain",
            details={"error": str(e)}
        )

from typing import Generator
from langchain.callbacks.base import BaseCallbackHandler
from flask import Response
import json

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses"""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Handle new tokens as they are generated"""
        self.tokens.append(token)
        # Yield the token as a server-sent event
        chunk = json.dumps({"status": "streaming", "token": token})
        return f"data: {chunk}\n\n"

def format_response(similar_docs, user_input, conversation) -> Generator:
    """
    Generate a streaming response using the conversation chain and similar documents.
    
    Args:
        similar_docs (list): List of similar documents from vector store
        user_input (str): User's question or message
        conversation: The conversation chain object
        
    Returns:
        Generator: Yields response tokens as they are generated
    """
    try:
        # Extract relevant context from similar documents
        context = "\n".join([doc.page_content for doc in similar_docs])
        
        # Create streaming callback handler
        handler = StreamingCallbackHandler()
        conversation.llm.callbacks = [handler]
        
        # Start the response stream
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"
        
        try:
            # Generate response with streaming
            for chunk in conversation.predict(
                context=context,
                input=user_input
            ):
                yield handler.on_llm_new_token(chunk)
            
            # End the stream
            yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            error_msg = "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
            yield "data: " + json.dumps({"status": "error", "message": error_msg}) + "\n\n"
            
    except Exception as e:
        # Log the error for setup issues
        logger.error(f"Error setting up response generation: {str(e)}")
        
        # Return error message as a server-sent event
        error_msg = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
        yield "data: " + json.dumps({"status": "error", "message": error_msg}) + "\n\n"

def validate_input(user_input: str) -> tuple[bool, str]:
    """
    Validate and sanitize user input.
    
    Args:
        user_input (str): User's input message
        
    Returns:
        tuple: (is_valid, sanitized_input)
        
    Raises:
        ValidationError: If input validation fails
    """
    try:
        if not user_input:
            raise ValidationError("Input cannot be empty")
            
        if not isinstance(user_input, str):
            raise ValidationError(
                "Invalid input type",
                details={"expected": "string", "received": type(user_input).__name__}
            )
        
        # Remove any potentially harmful characters and whitespace
        sanitized = user_input.strip()
        
        # Check input length
        if len(sanitized) < 2:
            raise ValidationError(
                "Input too short",
                details={"min_length": 2, "received_length": len(sanitized)}
            )
            
        if len(sanitized) > 500:
            raise ValidationError(
                "Input too long",
                details={"max_length": 500, "received_length": len(sanitized)}
            )
            
        # Log successful validation
        logger.debug(f"Input validated successfully: {sanitized[:50]}...")
        
        return True, sanitized
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            "Unexpected error during input validation",
            details={"error": str(e)}
        )
