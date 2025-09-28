from openai import OpenAI
import os
import logging
import json
import time
from typing import Generator, Optional, Dict, Any, List, Tuple
from .error_handler import APIError, ValidationError, ConfigurationError
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configure logging with more appropriate level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.messages = []
        self.max_history = max_history
        self.user_info = {}
        self.session_start = datetime.now()
        
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Keep only the last max_history messages (excluding system message)
        if len(self.messages) > self.max_history + 1:  # +1 for system message
            # Keep system message and recent messages
            system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
            recent_messages = self.messages[-(self.max_history):]
            self.messages = ([system_msg] if system_msg else []) + recent_messages
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API calls."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def update_user_info(self, key: str, value: Any):
        """Update user information for personalization."""
        self.user_info[key] = value
    
    def get_context_summary(self) -> str:
        """Generate a context summary for the current conversation."""
        if len(self.messages) <= 1:  # Only system message
            return "This is the beginning of our conversation."
        
        recent_topics = []
        for msg in self.messages[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                content_lower = msg["content"].lower()
                if any(keyword in content_lower for keyword in ['anxiety', 'anxious', 'panic']):
                    recent_topics.append("anxiety")
                elif any(keyword in content_lower for keyword in ['depression', 'depressed', 'sad']):
                    recent_topics.append("depression")
                elif any(keyword in content_lower for keyword in ['stress', 'overwhelmed']):
                    recent_topics.append("stress")
                elif any(keyword in content_lower for keyword in ['sleep', 'insomnia']):
                    recent_topics.append("sleep issues")
        
        if recent_topics:
            return f"We've been discussing: {', '.join(set(recent_topics))}."
        return "We've been having a supportive conversation."


class PersonalizedMentalHealthBot:
    """Enhanced mental health chatbot with OpenRouter API and conversation memory."""
    
    def __init__(self):
        self.client = None
        self.memory = ConversationMemory()
        self._initialize_client()
        self._setup_system_prompt()
    
    def _initialize_client(self):
        """Initialize OpenRouter client."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY environment variable is not set",
                details={"required_env_var": "OPENROUTER_API_KEY"}
            )
        
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise ConfigurationError(f"Failed to initialize OpenRouter client: {e}")
    
    def _setup_system_prompt(self):
        """Setup the personalized system prompt."""
        system_prompt = """You are Alex, a warm, empathetic, and highly trained mental health support companion. You're here to provide a safe, non-judgmental space where people can share their thoughts and feelings.

Your approach:
• **Listen deeply**: Always acknowledge what the person has shared and reflect back their emotions
• **Ask thoughtful questions**: Help them explore their feelings and thoughts more deeply
• **Be genuinely curious**: Show interest in their experiences, challenges, and what matters to them
• **Offer personalized support**: Remember what they've shared and reference it in future responses
• **Validate their experiences**: Make it clear that their feelings are valid and understandable
• **Be conversational**: Respond naturally, like a caring friend who happens to be a mental health professional

Your personality:
• Warm and approachable, yet professional
• Patient and never rushed
• Genuinely interested in their well-being
• Encouraging without being dismissive of their struggles
• Culturally sensitive and inclusive

Remember:
• Always start by acknowledging their current emotional state
• Ask follow-up questions to better understand their situation
• Share relevant coping strategies only after understanding their specific context
• If they mention concerning thoughts about self-harm, gently but clearly encourage them to seek immediate professional help
• Keep responses conversational - avoid bullet points unless specifically requested
• Reference previous parts of your conversation when relevant to show you're truly listening

Your goal is to make each person feel heard, understood, and supported while providing helpful mental health guidance."""

        self.memory.add_message("system", system_prompt)
    
    def generate_response(self, user_input: str, similar_docs: List = None) -> str:
        """Generate a personalized response using OpenRouter API."""
        try:
            # Validate input
            is_valid, sanitized_input = validate_input(user_input)
            
            # Add user message to memory
            self.memory.add_message("user", sanitized_input)
            
            # Extract any relevant context from documents
            context = self._extract_context_from_docs(similar_docs) if similar_docs else ""
            
            # Create enhanced user message with context
            enhanced_input = self._enhance_input_with_context(sanitized_input, context)
            
            # Update the last user message with enhanced content
            if self.memory.messages and self.memory.messages[-1]["role"] == "user":
                self.memory.messages[-1]["content"] = enhanced_input
            
            # Get messages for API
            messages = self.memory.get_messages_for_api()
            
            # Make API call with retry logic
            response = self._make_api_call_with_retry(messages)
            
            # Add assistant response to memory
            self.memory.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Provide a fallback response
            fallback_response = self._generate_fallback_response(user_input)
            self.memory.add_message("assistant", fallback_response)
            return fallback_response
    
    def _enhance_input_with_context(self, user_input: str, context: str) -> str:
        """Enhance user input with relevant context from documents."""
        if not context or "No specific context available" in context:
            return user_input
        
        return f"{user_input}\n\n[Context from similar discussions: {context[:200]}...]"
    
    def _extract_context_from_docs(self, similar_docs: List) -> str:
        """Extract relevant context from similar documents."""
        if not similar_docs:
            return ""
        
        context_parts = []
        for doc in similar_docs[:2]:  # Limit to top 2 documents
            if hasattr(doc, 'page_content'):
                # Extract key insights rather than full content
                content = doc.page_content[:150]  # First 150 chars
                context_parts.append(content)
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _make_api_call_with_retry(self, messages: List[Dict], max_retries: int = 3) -> str:
        """Make API call with retry logic."""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="google/gemini-2.5-flash-preview-09-2025",
                    messages=messages,
                    temperature=0.8,  # Slightly creative but focused
                    max_tokens=500,   # Reasonable response length
                    top_p=0.9
                )
                
                response = completion.choices[0].message.content
                if response and response.strip():
                    return response.strip()
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise APIError(f"Failed to get response after {max_retries} attempts: {e}")
        
        return self._generate_fallback_response("")
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate a personalized fallback response when API fails."""
        context_summary = self.memory.get_context_summary()
        
        fallback_responses = [
            f"I'm experiencing some technical difficulties right now, but I want you to know that I'm still here with you. {context_summary} How are you feeling in this moment?",
            
            f"I apologize for the technical hiccup. What you're sharing is important to me. {context_summary} Can you tell me more about what's on your mind right now?",
            
            f"Even though I'm having some connection issues, I want to make sure you feel heard. {context_summary} What would be most helpful for you to talk about right now?",
        ]
        
        # Choose response based on conversation history
        if len(self.memory.messages) > 3:
            return fallback_responses[0]
        elif any("anxiety" in msg["content"].lower() for msg in self.memory.messages[-3:] if msg["role"] == "user"):
            return "I can see you've been sharing about some anxiety you're experiencing. Even though I'm having technical difficulties, I want you to know that what you're going through is valid. Can you tell me what's feeling most overwhelming right now?"
        else:
            return fallback_responses[1]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "message_count": len(self.memory.messages),
            "session_duration": str(datetime.now() - self.memory.session_start),
            "context_summary": self.memory.get_context_summary(),
            "user_info": self.memory.user_info
        }


# Global bot instance
_bot_instance = None

def get_conversation_chain():
    """
    Get the personalized mental health bot instance.
    
    Returns:
        PersonalizedMentalHealthBot: The configured bot instance
        
    Raises:
        ConfigurationError: If bot initialization fails
    """
    global _bot_instance
    
    if _bot_instance is None:
        try:
            _bot_instance = PersonalizedMentalHealthBot()
            logger.info("Personalized mental health bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            raise ConfigurationError(f"Failed to initialize mental health bot: {e}")
    
    return _bot_instance


def format_response(similar_docs: List, user_input: str, chain=None) -> Generator:
    """
    Generate a streaming response using the personalized bot.
    
    Args:
        similar_docs: List of similar documents from vector store
        user_input: User's question or message
        chain: Bot instance (will be created if None)
        
    Yields:
        Server-sent event formatted responses
    """
    try:
        # Get bot instance
        if chain is None:
            chain = get_conversation_chain()
        
        if not isinstance(chain, PersonalizedMentalHealthBot):
            raise ConfigurationError("Invalid chain type")
        
        # Start streaming
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"
        
        # Generate response
        response = chain.generate_response(user_input, similar_docs)
        
        # Stream the response
        yield from _stream_response(response)
        
        # Complete the stream
        yield "data: " + json.dumps({
            "status": "complete", 
            "conversation_summary": chain.get_conversation_summary()
        }) + "\n\n"
        
    except Exception as e:
        logger.error(f"Error in format_response: {str(e)}")
        error_msg = "I'm having some technical difficulties, but I'm still here to support you. Please try again."
        yield "data: " + json.dumps({"status": "error", "message": error_msg}) + "\n\n"


def _stream_response(response: str) -> Generator:
    """Stream response as server-sent events with natural typing simulation."""
    # Split into sentences for more natural streaming
    sentences = response.split('. ')
    
    for i, sentence in enumerate(sentences):
        if i < len(sentences) - 1:
            sentence += '. '
        
        # Stream words within each sentence
        words = sentence.split()
        for word in words:
            chunk = json.dumps({"status": "streaming", "token": word + " "})
            yield f"data: {chunk}\n\n"
            time.sleep(0.05)  # Natural typing speed
        
        # Slight pause between sentences
        if i < len(sentences) - 1:
            time.sleep(0.1)


def validate_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate and sanitize user input.
    
    Args:
        user_input: User's input message
        
    Returns:
        Tuple of (is_valid, sanitized_input)
        
    Raises:
        ValidationError: If input validation fails
    """
    if not user_input:
        raise ValidationError("Input cannot be empty")
    
    if not isinstance(user_input, str):
        raise ValidationError(
            "Invalid input type",
            details={"expected": "string", "received": type(user_input).__name__}
        )
    
    # Sanitize input
    sanitized = user_input.strip()
    
    # Validate length
    if len(sanitized) < 1:
        raise ValidationError(
            "Input too short",
            details={"min_length": 1, "received_length": len(sanitized)}
        )
    
    if len(sanitized) > 1000:  # Increased limit for more natural conversations
        raise ValidationError(
            "Input too long", 
            details={"max_length": 1000, "received_length": len(sanitized)}
        )
    
    return True, sanitized


def reset_conversation():
    """Reset the conversation memory for a new session."""
    global _bot_instance
    if _bot_instance:
        _bot_instance.memory = ConversationMemory()
        _bot_instance._setup_system_prompt()
        logger.info("Conversation reset successfully")


def get_conversation_history() -> List[Dict]:
    """Get the current conversation history."""
    global _bot_instance
    if _bot_instance:
        return _bot_instance.memory.get_messages_for_api()
    return []