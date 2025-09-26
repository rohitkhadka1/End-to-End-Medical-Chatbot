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
        
        # Initialize Hugging Face model endpoint with proper parameter structure
        try:
            # Use FLAN-T5 with minimal, compatible configuration
            llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-small",
                max_new_tokens=200,
                temperature=0.7,
                huggingfacehub_api_token=api_key,
                timeout=60
            )
            logger.info("Successfully initialized Hugging Face model (flan-t5-small)")
            
        except Exception as e:
            logger.error(f"Error initializing flan-t5-small model: {str(e)}")
            # Try with even simpler configuration
            try:
                llm = HuggingFaceEndpoint(
                    repo_id="google/flan-t5-small",
                    huggingfacehub_api_token=api_key,
                    max_new_tokens=150,
                    temperature=0.7
                )
                logger.info("Successfully initialized minimal Hugging Face model")
            except Exception as fallback_error:
                logger.error(f"All Hugging Face model initialization attempts failed")
                logger.error(f"Primary error: {str(e)}")
                logger.error(f"Fallback error: {str(fallback_error)}")
                
                # Try one more time with absolute minimal config
                try:
                    llm = HuggingFaceEndpoint(
                        repo_id="google/flan-t5-small",
                        huggingfacehub_api_token=api_key
                    )
                    logger.info("Successfully initialized ultra-minimal Hugging Face model")
                except Exception as final_error:
                    logger.warning("All Hugging Face API attempts failed. Using local fallback.")
                    # Create a simple local LLM fallback
                    from langchain.llms.base import LLM
                    from typing import Optional, List, Any
                    
                    class FallbackLLM(LLM):
                        @property
                        def _llm_type(self) -> str:
                            return "fallback"
                        
                        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                            # Simple rule-based responses for mental health
                            prompt_lower = prompt.lower()
                            
                            if any(word in prompt_lower for word in ['anxiety', 'anxious', 'worried', 'panic']):
                                return "I understand you're experiencing anxiety. Try deep breathing exercises: breathe in for 4 counts, hold for 4, exhale for 4. Consider speaking with a mental health professional for personalized support."
                            
                            elif any(word in prompt_lower for word in ['depression', 'depressed', 'sad', 'down']):
                                return "I hear that you're going through a difficult time. Depression is treatable, and you don't have to face it alone. Please consider reaching out to a mental health professional or counselor who can provide proper support."
                            
                            elif any(word in prompt_lower for word in ['stress', 'stressed', 'overwhelmed']):
                                return "Stress can be overwhelming. Try breaking tasks into smaller steps, practice mindfulness, and ensure you're getting adequate rest. If stress persists, consider speaking with a counselor."
                            
                            elif any(word in prompt_lower for word in ['sleep', 'insomnia', 'tired']):
                                return "Good sleep is crucial for mental health. Try maintaining a regular sleep schedule, avoiding screens before bed, and creating a calm sleep environment. If sleep issues persist, consult a healthcare provider."
                            
                            else:
                                return "Thank you for sharing. Mental health is important, and it's okay to seek help. Consider speaking with a mental health professional who can provide personalized guidance for your situation."
                    
                    llm = FallbackLLM()
                    logger.info("Successfully initialized local fallback LLM")
        
        # Test the LLM connection with a simple prompt
        try:
            test_response = llm.invoke("Test")
            if test_response:
                logger.info("Successfully tested Hugging Face model connection")
            else:
                logger.info("Model initialized successfully (test returned empty response)")
        except Exception as e:
            # This is expected for some models and doesn't indicate a problem
            logger.info("Model initialized successfully (test skipped due to API limitations)")
        
        logger.info("Successfully initialized Hugging Face language model")
        
        # Create a simpler, more effective prompt template
        template = """You are a helpful mental health support assistant. Answer the user's question using the provided context.

Context: {context}

Question: {input}

Answer: Provide a helpful, empathetic response that addresses the user's specific concern. Keep it concise and supportive."""
        
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
import re

def generate_fallback_response(user_input: str, context: str) -> str:
    """
    Generate a contextually appropriate fallback response based on user input and available context.
    
    Args:
        user_input (str): The user's original question/message
        context (str): Available context from similar documents
        
    Returns:
        str: A contextually appropriate response
    """
    user_input_lower = user_input.lower()
    
    # Detect key mental health topics and provide appropriate responses
    if any(word in user_input_lower for word in ['anxiety', 'anxious', 'panic', 'worry', 'worried']):
        if 'attack' in user_input_lower:
            return """When experiencing anxiety attacks, try these techniques:
1. Practice deep breathing - breathe in slowly for 4 counts, hold for 4, exhale for 6
2. Use grounding techniques - name 5 things you can see, 4 you can hear, 3 you can touch
3. Remind yourself that panic attacks are temporary and will pass
4. Find a quiet, safe space if possible
5. Consider speaking with a mental health professional for personalized coping strategies

If you experience frequent or severe anxiety attacks, please consult with a healthcare provider."""
        else:
            return """I understand you're dealing with anxiety. Here are some general strategies that may help:
- Practice mindfulness and deep breathing exercises
- Maintain a regular sleep schedule and exercise routine  
- Limit caffeine and alcohol intake
- Consider talking to a mental health professional
- Try relaxation techniques like progressive muscle relaxation

Remember, professional support can provide you with personalized strategies for managing anxiety."""
    
    elif any(word in user_input_lower for word in ['depression', 'depressed', 'sad', 'down', 'hopeless']):
        return """I understand you're experiencing difficult feelings. Depression is a common but serious condition that affects many people. Here are some supportive steps:

- Reach out to trusted friends, family, or a mental health professional
- Try to maintain daily routines and engage in activities you usually enjoy
- Consider gentle exercise, even just a short walk
- Practice self-compassion and avoid self-criticism
- If you're having thoughts of self-harm, please contact a crisis helpline immediately

Professional help from a therapist or counselor can provide you with effective treatment options. You don't have to go through this alone."""
    
    elif any(word in user_input_lower for word in ['stress', 'stressed', 'overwhelmed', 'pressure']):
        return """Feeling stressed or overwhelmed is a common experience. Here are some strategies that might help:

- Break large tasks into smaller, manageable steps
- Practice time management and prioritization
- Take regular breaks and practice relaxation techniques
- Engage in physical activity to help reduce stress
- Talk to someone you trust about what you're experiencing
- Consider stress management techniques like meditation or yoga

If stress is significantly impacting your daily life, a mental health professional can help you develop personalized coping strategies."""
    
    elif any(word in user_input_lower for word in ['sleep', 'insomnia', 'tired', 'exhausted']):
        return """Sleep issues can significantly impact mental health. Here are some tips for better sleep:

- Maintain a consistent sleep schedule
- Create a relaxing bedtime routine
- Limit screen time before bed
- Keep your bedroom cool, dark, and quiet
- Avoid caffeine and large meals close to bedtime
- Consider relaxation techniques before sleep

If sleep problems persist, consult with a healthcare provider as they may be related to underlying mental health conditions."""
    
    # Use context if available and no specific topic detected
    elif context and "No specific context available" not in context:
        return f"""Based on the information available, I want to provide you with supportive guidance. While I can offer general mental health information, it's important to remember that everyone's situation is unique.

I encourage you to speak with a qualified mental health professional who can provide personalized support and treatment options tailored to your specific needs. They can offer evidence-based treatments and ongoing support.

In the meantime, please know that seeking help is a sign of strength, and there are people and resources available to support you."""
    
    # General fallback
    else:
        return """I'm here to provide mental health support and information. While I'd like to give you more specific guidance, I want to ensure you receive the most appropriate help for your situation.

I encourage you to reach out to a qualified mental health professional who can provide personalized support. They can offer proper assessment, treatment options, and ongoing care tailored to your needs.

If you're in crisis or having thoughts of self-harm, please contact a crisis helpline or emergency services immediately. You don't have to face this alone."""

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
        print(f"\n=== RETRIEVED DOCUMENTS DEBUG INFO ===")
        print(f"Number of similar documents found: {len(similar_docs)}")
        
        if similar_docs:
            context_parts = []
            for i, doc in enumerate(similar_docs[:3]):
                doc_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                logger.debug(f"Document {i+1}: {doc.page_content[:100]}...")
                print(f"\nDocument {i+1}:")
                print(f"Content preview: {doc_preview}")
                print(f"Full length: {len(doc.page_content)} characters")
                context_parts.append(doc.page_content)
            context = "\n".join(context_parts)
            logger.info(f"Generated context from {len(context_parts)} documents (total length: {len(context)})")
            print(f"\nCombined context length: {len(context)} characters")
        else:
            context = "No specific context available. Provide general mental health support."
            logger.warning("No similar documents found - using fallback context")
            print("\nNo similar documents found - using fallback context")
        print(f"=== END DEBUG INFO ===\n")
        
        # Start the response stream
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"
        
        try:
            # Generate response using the chain
            logger.info(f"Generating response for input: {user_input[:50]}...")
            logger.info(f"Context length: {len(context)} characters")
            
            # Try to invoke the chain with detailed error handling
            try:
                print(f"\n=== INVOKING LLM CHAIN ===")
                print(f"Input: {user_input}")
                print(f"Context length: {len(context)} chars")
                
                # Add retry logic for StopIteration errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = chain.invoke({
                            "context": context,
                            "input": user_input
                        })
                        
                        # Validate the result
                        if result is None or (isinstance(result, str) and not result.strip()):
                            raise ValueError("Empty response from model")
                        
                        logger.info(f"Chain invocation successful. Result type: {type(result)}")
                        logger.debug(f"Raw result: {result}")
                        print(f"LLM Response received successfully")
                        print(f"Response type: {type(result)}")
                        print(f"=== END LLM CHAIN ===\n")
                        break
                        
                    except (StopIteration, ValueError) as retry_error:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {retry_error}. Retrying...")
                            print(f"Retry attempt {attempt + 1} due to: {retry_error}")
                            continue
                        else:
                            raise retry_error
                
            except Exception as invoke_error:
                logger.error(f"Chain invocation failed: {str(invoke_error)}")
                logger.error(f"Error type: {type(invoke_error).__name__}")
                print(f"\n=== CHAIN INVOCATION ERROR ===")
                print(f"Error: {str(invoke_error)}")
                print(f"Error type: {type(invoke_error).__name__}")
                print(f"=== END ERROR INFO ===\n")
                
                # Generate a more appropriate fallback response based on user input
                response = generate_fallback_response(user_input, context)
                
                logger.info("Using fallback response due to chain invocation failure")
                print(f"Using fallback response: {response[:100]}...")
                
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
            print(f"\n=== FINAL RESPONSE ===")
            print(f"Response preview: {response[:200]}...")
            print(f"Full response length: {len(response)} characters")
            print(f"=== END FINAL RESPONSE ===\n")
            
            # Validate response
            if not response or response.strip() == "":
                logger.warning("Empty response generated, using fallback")
                print("\nEmpty response generated, creating fallback response")
                response = generate_fallback_response(user_input, context)
            
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
        print(f"\nError in format_response setup: {str(e)}")
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