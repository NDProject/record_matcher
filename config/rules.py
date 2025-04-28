"""Column addition rules for the record matching system."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from dataclasses import dataclass
import regex as re

class ColumnRule(ABC):
    """Base class for column selection rules."""
    
    @abstractmethod
    def should_add_column(self, column_name: str, source_columns: List[str]) -> bool:
        """
        Determine if a column should be added to the result.

        Args:
            column_name: Name of the column from the match dataset
            source_columns: List of columns from the source dataset
        
        Returns:
            bool: Whether the column should be added
        """
        pass

class NewColumnsRule(ColumnRule):
    """Select columns that don't exist in source dataset."""
    
    def should_add_column(self, column_name: str, source_columns: List[str]) -> bool:
        return column_name not in source_columns

class PatternRule(ColumnRule):
    """Select columns matching a regex pattern."""
    
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)
    
    def should_add_column(self, column_name: str, source_columns: List[str]) -> bool:
        return bool(self.pattern.match(column_name))

@dataclass
class AdditionRules:
    """Configuration for which columns to add from matches."""
    
    include_rules: List[ColumnRule]
    exclude_columns: Optional[List[str]] = None
    
    def should_add_column(self, column_name: str, source_columns: List[str]) -> bool:
        """
        Determine if a column should be added based on all rules.
        
        Args:
            column_name: Name of the column from match dataset
            source_columns: List of columns from source dataset
            
        Returns:
            bool: Whether the column should be added
        """
        if self.exclude_columns and column_name in self.exclude_columns:
            return False
            
        return any(
            rule.should_add_column(column_name, source_columns)
            for rule in self.include_rules
        )