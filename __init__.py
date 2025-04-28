"""
Record Matcher
=============

A flexible and efficient system for matching records across datasets using
multiple matching strategies including TF-IDF analysis, fuzzy matching, and 
configurable validation rules.

Key Features:
- Multiple matching strategies with configurable thresholds
- TF-IDF analysis for improved name matching
- Flexible preprocessing with date handling
- Parallel processing support
- Detailed match analysis and reporting
"""

from core.matcher import RecordMatcher

from config.models import (
    MatchStrategy,
    ColumnMatchConfig,
    TFIDFConfig,
    PartialMatchConfig
)
from config.rules import AdditionRules, ColumnRule

__version__ = "1.0.0"