"""
Logging module for ORVIXA.
Provides structured logging throughout the application.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

class ORVIXALogger:
    """Centralized logging system for ORVIXA."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self):
        """Initialize the logger with file and console handlers."""
        self._logger = logging.getLogger('ORVIXA')
        self._logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"orvixa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get the logger instance."""
        return self._logger


def get_logger():
    """Get a logger instance."""
    return ORVIXALogger().get_logger()
