from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

def get_conversation_chain():
    """
    Create a conversation chain with memory and custom prompt template.
    Returns the conversation chain object.
    """
    # Initialize language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create a custom prompt template for mental health conversations
    template = """You are a helpful and empathetic mental health assistant. You use the provided context to offer support 
    and information based on mental health best practices. Always maintain a professional and supportive tone.
    
    Context from mental health manual: {context}
    
    Current conversation:
    {history}
    Human: {input}
    Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "history", "input"],
        template=template
    )
    
    # Set up conversation memory
    memory = ConversationBufferMemory(
        return_messages=True,
        human_prefix="Human",
        ai_prefix="Assistant"
    )
    
    # Create the conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    
    return conversation

def format_response(similar_docs, user_input, conversation):
    """
    Generate a response using the conversation chain and similar documents.
    
    Args:
        similar_docs (list): List of similar documents from vector store
        user_input (str): User's question or message
        conversation: The conversation chain object
        
    Returns:
        str: Generated response
    """
    # Extract relevant context from similar documents
    context = "\n".join([doc.page_content for doc in similar_docs])
    
    # Get response from conversation chain
    response = conversation.predict(
        context=context,
        input=user_input
    )
    
    return response

def validate_input(user_input):
    """
    Validate and sanitize user input.
    
    Args:
        user_input (str): User's input message
        
    Returns:
        tuple: (is_valid, sanitized_input)
    """
    if not user_input or not isinstance(user_input, str):
        return False, ""
    
    # Remove any potentially harmful characters
    sanitized = user_input.strip()
    
    # Check if the input is too long or too short
    if len(sanitized) < 2 or len(sanitized) > 500:
        return False, ""
        
    return True, sanitized
