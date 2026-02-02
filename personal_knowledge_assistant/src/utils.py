"""
Utility Functions - Logging and text cleaning
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging for the application."""
    logger = logging.getLogger("knowledge_assistant")
    logger.handlers = []
    logger.setLevel(getattr(logging, log_level))
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


logger = setup_logging()


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text while preserving structure."""
    # Remove invalid characters FIRST
    text = text.replace("\x00", "").replace("\ufffd", "")
    
    # Preserve paragraph breaks by normalizing line-by-line
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(line for line in lines if line.strip())
    
    # Final normalization
    return text.strip()
