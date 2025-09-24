from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnableSequence
import os
import logging
from typing import Generator, Optional, Dict, Any
from .error_handler import APIError, ValidationError, ConfigurationError
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOutputParser(BaseOutputParser):
    """Simple output parser that returns the text as-is."""
    
    def parse(self, text: str) -> str:
        return text.strip()

def get_conversation_chain() -> RunnableSequence:
    """
    Create a simple runnable chain with custom prompt template using Hugging Face models.
    Returns the runnable chain object.
    
    Raises:
        APIError: If there's an error initializing the language model
    """
    try:
        # Initialize Hugging Face API key
        api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not api_key:
            raise ConfigurationError(
                "HUGGINGFACEHUB_API_TOKEN environment variable is not set",
                details={"required_env_var": "HUGGINGFACEHUB_API_TOKEN"}
            )
        # Set the environment variable for Hugging Face
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        
        # Initialize Hugging Face model endpoint with a reliable model
        try:
            # Try with simpler parameters first
            llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-small",  # Start with smaller, more reliable model
                max_new_tokens=256,
                temperature=0.7,
                huggingfacehub_api_token=api_key,
                timeout=30,  # Shorter timeout
            )
            logger.info("Successfully initialized Hugging Face model (flan-t5-small)")
            
        except Exception as e:
            logger.error(f"Error initializing Hugging Face model: {str(e)}")
            # Try even simpler configuration
            try:
                llm = HuggingFaceEndpoint(
                    repo_id="google/flan-t5-small",
                    huggingfacehub_api_token=api_key,
                    timeout=30,
                )
                logger.info("Successfully initialized fallback Hugging Face model (minimal config)")
            except Exception as fallback_error:
                logger.error(f"All Hugging Face model initialization attempts failed")
                logger.error(f"Primary error: {str(e)}")
                logger.error(f"Fallback error: {str(fallback_error)}")
                raise APIError(
                    "Failed to initialize any Hugging Face model",
                    details={"primary_error": str(e), "fallback_error": str(fallback_error)}
                )
        
        # Test the LLM connection
        try:
            test_response = llm.invoke("Hello")
            logger.info("Successfully tested Hugging Face model connection")
        except Exception as e:
            logger.warning(f"Model test failed but continuing: {str(e)}")
        
        logger.info("Successfully initialized Hugging Face language model")
        
        # Create a custom prompt template for mental health conversations
        template = """You are a helpful and empathetic mental health assistant. Provide supportive guidance while maintaining professional boundaries.

Guidelines:
- Be compassionate and supportive
- Provide general information, not diagnoses
- Suggest professional help when needed
- Use respectful, inclusive language
- Keep responses concise and helpful

Context from knowledge base: {context}

User question: {input}

Provide a helpful and supportive response:"""
        
        # Create the prompt template with input validation
        try:
            prompt = PromptTemplate(
                input_variables=["context", "input"],
                template=template,
                validate_template=True
            )
        except Exception as e:
            raise ConfigurationError(
                "Error creating prompt template",
                details={"error": str(e)}
            )
        
        # Create output parser
        output_parser = SimpleOutputParser()
        
        # Create the runnable chain with comprehensive configuration
        try:
            # Create a modern runnable sequence: prompt | llm | output_parser
            chain = prompt | llm | output_parser
            
            # Verify the chain was created successfully
            if chain is None:
                raise ConfigurationError(
                    "Runnable chain creation returned None",
                    details={"llm_type": type(llm).__name__, "prompt_vars": prompt.input_variables}
                )
            
            logger.info("Runnable chain created successfully")
            return chain
            
        except Exception as e:
            logger.error(f"Detailed chain creation error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise ConfigurationError(
                "Error creating runnable chain",
                details={"error": str(e), "error_type": type(e).__name__}
            )
            
    except (ConfigurationError, APIError) as e:
        logger.error(f"Configuration/API error in get_conversation_chain: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during chain initialization: {str(e)}")
        raise ConfigurationError(
            "Failed to initialize runnable chain",
            details={"error": str(e)}
        )

import json
import time

def format_response(similar_docs, user_input, chain) -> Generator:
    """
    Generate a streaming response using the LLM chain and similar documents.
    
    Args:
        similar_docs (list): List of similar documents from vector store
        user_input (str): User's question or message
        chain: The LLM chain object
        
    Returns:
        Generator: Yields response tokens as they are generated
    """
    try:
        # Defensive check: ensure chain is initialized
        if chain is None:
            logger.error("LLM chain is None before generating a response")
            raise ConfigurationError(
                "LLM chain not initialized",
                details={"hint": "Ensure get_conversation_chain() succeeds before calling format_response"}
            )
        
        # Extract relevant context from similar documents
        logger.info(f"Processing {len(similar_docs)} similar documents")
        
        if similar_docs:
            context_parts = []
            for i, doc in enumerate(similar_docs[:3]):
                logger.debug(f"Document {i+1}: {doc.page_content[:100]}...")
                context_parts.append(doc.page_content)
            context = "\n".join(context_parts)
            logger.info(f"Generated context from {len(context_parts)} documents (total length: {len(context)})")
        else:
            context = "No specific context available. Provide general mental health support."
            logger.warning("No similar documents found - using fallback context")
        
        # Start the response stream
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"
        
        try:
            # Generate response using the chain
            logger.info(f"Generating response for input: {user_input[:50]}...")
            logger.info(f"Context length: {len(context)} characters")
            
            # Try to invoke the chain with detailed error handling
            try:
                result = chain.invoke({
                    "context": context,
                    "input": user_input
                })
                logger.info(f"Chain invocation successful. Result type: {type(result)}")
                logger.debug(f"Raw result: {result}")
                
            except Exception as invoke_error:
                logger.error(f"Chain invocation failed: {str(invoke_error)}")
                logger.error(f"Error type: {type(invoke_error).__name__}")
                
                # Fallback response
                response = f"I understand you're feeling depressed. That's a difficult experience, and I want you to know that you're not alone. Depression is a common mental health condition that affects many people. It's important to reach out for professional help from a mental health provider who can offer proper support and treatment options. In the meantime, please consider talking to someone you trust about how you're feeling."
                
                logger.info("Using fallback response due to chain invocation failure")
                
                # Send fallback response
                words = response.split()
                for i, word in enumerate(words):
                    chunk = json.dumps({"status": "streaming", "token": word + " "})
                    yield f"data: {chunk}\n\n"
                    time.sleep(0.03)
                
                yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
                return
            
            # Extract the response text
            if isinstance(result, dict):
                response = result.get("text", str(result))
            else:
                response = str(result)
            
            logger.info(f"Generated response: {response[:100]}...")
            
            # Validate response
            if not response or response.strip() == "":
                logger.warning("Empty response generated, using fallback")
                response = "I understand you're reaching out about mental health concerns. While I'd like to provide specific guidance, I'm experiencing some technical difficulties right now. Please consider speaking with a mental health professional who can provide you with proper support and guidance."
            
            # Send response in chunks to simulate streaming
            words = response.split()
            for i, word in enumerate(words):
                chunk = json.dumps({"status": "streaming", "token": word + " "})
                yield f"data: {chunk}\n\n"
                time.sleep(0.03)  # Simulate typing effect
            
            # End the stream
            yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
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
        
        # Check input length (adjusted for HF models)
        if len(sanitized) < 2:
            raise ValidationError(
                "Input too short",
                details={"min_length": 2, "received_length": len(sanitized)}
            )
            
        if len(sanitized) > 300:  # Reduced for better HF model performance
            raise ValidationError(
                "Input too long",
                details={"max_length": 300, "received_length": len(sanitized)}
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
print("Success")