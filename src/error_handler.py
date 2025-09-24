"""Error handling utilities for the chatbot application."""
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Base exception class for chatbot errors."""
    def __init__(self, message: str, error_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        
        # Log the error
        logger.error(f"{error_type}: {message}", extra={"details": details})

class APIError(ChatbotError):
    """Raised when there's an error with API calls."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)

class ValidationError(ChatbotError):
    """Raised when there's an error with input validation."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

class ConfigurationError(ChatbotError):
    """Raised when there's an error with configuration."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)

# Retry decorator for API calls
def retry_with_exponential_backoff(max_attempts: int = 3, max_wait: float = 10):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts (int): Maximum number of retry attempts
        max_wait (float): Maximum wait time between retries in seconds
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, max=max_wait),
        reraise=True,
        retry_error_callback=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}"
        )
    )

def format_error_response(error: Exception) -> Dict[str, str]:
    """
    Format an error for API response.
    
    Args:
        error: The exception to format
        
    Returns:
        Dict containing status and error message
    """
    if isinstance(error, ChatbotError):
        return {
            'status': 'error',
            'error_type': error.error_type,
            'message': str(error)
        }
    return {
        'status': 'error',
        'error_type': 'UNKNOWN_ERROR',
        'message': 'An unexpected error occurred. Please try again later.'
    }