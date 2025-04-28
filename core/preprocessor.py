"""Flexible and extensible preprocessing system for record matching."""

from typing import Any, Callable, Optional, Protocol, Dict, Type
from abc import ABC, abstractmethod
import pandas as pd
import re
import unicodedata
import logging
from functools import lru_cache
from datetime import datetime
from dateutil import parser
from config.models import DateConfig

class Preprocessor(Protocol):
    """Protocol defining the interface for preprocessors."""
    def process(self, value: Any) -> str:
        """Process a value into a standardized string format."""
        ...

class BasePreprocessor(ABC):
    """Base class for preprocessors with common functionality."""
    
    @abstractmethod
    def process(self, value: Any) -> str:
        """Process a value into a standardized string format."""
        pass

    def _handle_null(self, value: Any) -> bool:
        """Check if value is null/empty."""
        return pd.isna(value)

class NamePreprocessor(BasePreprocessor):
    """Preprocesses names with space-aware normalization."""
    
    def __init__(self, lowercase: bool = True, remove_accents: bool = True):
        self.lowercase = lowercase
        self.remove_accents = remove_accents
    
    def process(self, value: Any) -> str:
        if self._handle_null(value):
            return ''
            
        try:
            text = str(value)
            
            if self.lowercase:
                text = text.lower()
            
            if self.remove_accents:
                text = ''.join(
                    c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn'
                )
            
            # Replace non-alphanumeric chars with spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Create two versions
            spaced = ' '.join(text.split())
            compressed = ''.join(text.split())
            
            return f"{spaced}|{compressed}"
            
        except Exception as e:
            logging.warning(f"Error in name preprocessing: {e}")
            return ''

class PostalCodePreprocessor(BasePreprocessor):
    """Preprocesses postal codes."""
    
    def __init__(self, uppercase: bool = True):
        self.uppercase = uppercase
    
    def process(self, value: Any) -> str:
        if self._handle_null(value):
            return ''
            
        try:
            text = str(value)
            if self.uppercase:
                text = text.upper()
            return re.sub(r'\s+', '', text).strip()
        except Exception as e:
            logging.warning(f"Error in postal code preprocessing: {e}")
            return ''

class DatePreprocessor(BasePreprocessor):
    """Preprocesses dates with flexible configuration."""
    
    def __init__(self, config: DateConfig):
        self.config = config
        self._setup_parser()
    
    def _setup_parser(self) -> None:
        if isinstance(self.config.format, list):
            self._parse = self._parse_multiple_formats
            self._formats = self.config.format
        elif isinstance(self.config.format, str):
            self._parse = self._parse_single_format
            self._formats = [self.config.format]
        else:
            self._parse = self._parse_flexible
            self._formats = []
    
    def _parse_single_format(self, value: str) -> Optional[datetime]:
        try:
            return datetime.strptime(value, self._formats[0])
        except (ValueError, TypeError):
            return None
    
    def _parse_multiple_formats(self, value: str) -> Optional[datetime]:
        for fmt in self._formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
    
    def _parse_flexible(self, value: str) -> Optional[datetime]:
        try:
            return parser.parse(value)
        except (ValueError, TypeError):
            return None
    
    def process(self, value: Any) -> str:
        if self._handle_null(value):
            return self.config.invalid_value
            
        try:
            value_str = str(value).strip()
            dt = self._parse(value_str)
            
            if dt is None:
                return self.config.invalid_value
                
            # Apply normalization
            if self.config.normalize_to == 'month':
                dt = dt.replace(day=1)
            elif self.config.normalize_to == 'quarter':
                dt = dt.replace(
                    month=((dt.month - 1) // 3) * 3 + 1,
                    day=1
                )
                
            # Format output
            output_format = (
                "%Y-%m-%d %H:%M:%S" if self.config.include_time 
                else "%Y-%m-%d"
            )
            return dt.strftime(output_format)
            
        except Exception as e:
            logging.warning(f"Error in date preprocessing: {e}")
            return self.config.invalid_value

class PreprocessorRegistry:
    """Registry for preprocessor types and instances."""
    
    def __init__(self):
        self._preprocessors: Dict[str, Type[BasePreprocessor]] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default preprocessors."""
        self.register('name', NamePreprocessor)
        self.register('postal_code', PostalCodePreprocessor)
        self.register('date', DatePreprocessor)
    
    def register(self, name: str, preprocessor_class: Type[BasePreprocessor]) -> None:
        """
        Register a new preprocessor type.
        
        Args:
            name: Name to register the preprocessor under
            preprocessor_class: Preprocessor class to register
        """
        self._preprocessors[name] = preprocessor_class
    
    def create(
        self,
        name: str,
        **kwargs: Any
    ) -> BasePreprocessor:
        """
        Create a preprocessor instance.
        
        Args:
            name: Name of the preprocessor type
            **kwargs: Configuration parameters for the preprocessor
            
        Returns:
            BasePreprocessor: Configured preprocessor instance
            
        Raises:
            ValueError: If preprocessor type not found
        """
        preprocessor_class = self._preprocessors.get(name)
        if not preprocessor_class:
            raise ValueError(f"Unknown preprocessor type: {name}")
        
        return preprocessor_class(**kwargs)

# Global registry instance
registry = PreprocessorRegistry()

def register_preprocessor(name: str, preprocessor_class: Type[BasePreprocessor]) -> None:
    """
    Register a new preprocessor type globally.
    
    Args:
        name: Name to register the preprocessor under
        preprocessor_class: Preprocessor class to register
    """
    registry.register(name, preprocessor_class)