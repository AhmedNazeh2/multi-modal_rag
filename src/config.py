import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv(override=True)

class Config:
    """
    Configuration class for the application.
    Centralizes all configuration parameters and supports environment variable overrides.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    
    # Text chunking parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    WHISPER_MODEL = os.getenv("WHISPER_MODEL_NAME", "large-v3")
    SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
    
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") 
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") 
    # Retrieval settings
    TOP_K = int(os.getenv("TOP_K", "5"))
    
    # LLM parameters
    # MODIFIED: Changed default temperature to 0.2 for better RAG responses
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0")) 
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Memory settings
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))
    
    @classmethod
    def get_llm_params(cls) -> Dict[str, Any]:
        """
        Get LLM parameters as a dictionary for easy configuration.
        
        Returns:
            Dict[str, Any]: Dictionary of LLM parameters
        """
        return {
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS, 
        }
    
    @classmethod
    def is_valid(cls) -> bool:
        """
        Check if the configuration is valid (has required API keys).
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return bool(cls.OPENAI_API_KEY)