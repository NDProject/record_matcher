"""Configuration models for the record matching system."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import Enum
from datetime import datetime

@dataclass(frozen=True)
class TFIDFConfig:
    """Configuration for TF-IDF analysis."""
    enabled: bool = False
    min_df: float = 0.01  # Minimum document frequency (1%)
    max_df: float = 0.9   # Maximum document frequency (90%)
    ngram_range: Tuple[int, int] = (1, 3)
    analyzer: str = 'word'
    min_term_length: int = 4
    common_term_threshold: float = 5.0

@dataclass(frozen=True)
class PartialMatchConfig:
    """Configuration for partial string matching."""
    min_length: int
    required_match_percentage: float

class MatchCategory(str, Enum):
    """Categories for matching results."""
    EXACT = "exact"
    SIMILAR = "similar"
    NO_MATCH = "no_match"

@dataclass(frozen=True)
class DateConfig:
    """Configuration for date preprocessing."""
    format: Optional[Union[str, List[str]]] = None
    invalid_value: str = ''
    include_time: bool = False
    normalize_to: Optional[str] = None  # 'month', 'quarter', etc.

@dataclass(frozen=True)
class ColumnMatchConfig:
    """Configuration for how to match a specific column."""
    name: str
    weight: float = 1.0
    preprocess_method: str = 'name'
    match_method: str = 'exact'
    min_threshold: float = 1.0
    desired_threshold: Optional[float] = None
    is_required: bool = False
    min_content_length: int = 0
    strict_threshold: Optional[float] = None
    partial_match_config: Optional[PartialMatchConfig] = None
    tfidf_config: Optional[TFIDFConfig] = None
    date_config: Optional[DateConfig] = None

    def __post_init__(self):
        """Set default desired threshold if not provided."""
        object.__setattr__(
            self,
            'desired_threshold',
            self.desired_threshold or self.min_threshold
        )

@dataclass(frozen=True)
class MatchStrategy:
    """Strategy for matching records across datasets."""
    name: str
    column_configs: List[ColumnMatchConfig]
    min_threshold: float = 0.8
    desired_threshold: Optional[float] = None
    require_partial_validation: bool = False

    def __post_init__(self):
        """Set default desired threshold if not provided."""
        object.__setattr__(
            self,
            'desired_threshold',
            self.desired_threshold or self.min_threshold
        )

@dataclass
class PartialMatchResult:
    """Result of matching individual parts of strings."""
    source_part: str
    target_part: str
    similarity_score: float
    match_category: MatchCategory
    score_contribution: float
    position_index: Tuple[int, int]
    spaced_match: bool
    compressed_match: bool