"""TF-IDF analysis for improved text matching."""

import re
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import xxhash
from config.models import TFIDFConfig

@staticmethod
def _build_term_pattern(common_terms: Set[str]) -> re.Pattern:
    """
    Build regex pattern from common terms, sorted by length.
    Longest terms are matched first to avoid partial matches.
    """
    if not common_terms:
        return re.compile(r'$^')  # Match nothing
        
    # Sort terms by length (longest first) and escape special characters
    sorted_terms = sorted(common_terms, key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in sorted_terms]
    
    # Create pattern that matches whole words or parts of compound words
    pattern = r'|'.join(escaped_terms)
    return re.compile(pattern, re.IGNORECASE)

class TFIDFAnalyzer:
    """Analyzer for TF-IDF based text similarity with caching."""
    
    def __init__(
        self,
        config: TFIDFConfig,
        preprocessor: Optional[callable] = None
    ):
        """
        Initialize TF-IDF analyzer.
        
        Args:
            config: TF-IDF configuration
            preprocessor: Optional text preprocessing function
        """
        self.vectorizer = TfidfVectorizer(
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=config.ngram_range,
            analyzer=config.analyzer,
            preprocessor=preprocessor,
            token_pattern=r'[A-Za-z]' + f'{{{config.min_term_length},}}' + r'\w*',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        self.config = config
        self.common_terms: Set[str] = set()
        self.common_terms_pattern = None
        self.term_weights: Dict[str, float] = {}
        self.fitted = False
        self.feature_matrix = None
        self.texts_hash = None
        
        # Cache for similarity calculations
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.MAX_CACHE_SIZE = 10000
    
    def _compute_texts_hash(self, texts: List[str]) -> str:
        """Compute hash of input texts to detect changes."""
        return xxhash.xxh64(''.join(sorted(texts))).hexdigest()
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the analyzer to the input texts.
        
        Args:
            texts: List of texts to analyze
        """
        try:
            # Check if already fitted with same texts
            new_hash = self._compute_texts_hash(texts)
            if self.fitted and new_hash == self.texts_hash:
                return
                
            self.texts_hash = new_hash
            processed_texts = [str(text).lower() for text in texts if pd.notna(text)]
            
            # Fit and transform
            self.feature_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Calculate term statistics
            n_docs = len(processed_texts)
            term_doc_freq = np.bincount(
                self.feature_matrix.indices, 
                minlength=self.feature_matrix.shape[1]
            )
            
            feature_names = self.vectorizer.get_feature_names_out()
            avg_tfidf = np.asarray(self.feature_matrix.mean(axis=0)).ravel()
            
            self.term_weights = {}
            self.common_terms = set()
            
            # Process terms
            for idx, (term, freq, importance) in enumerate(zip(
                feature_names, term_doc_freq, avg_tfidf
            )):
                freq_pct = (freq / n_docs) * 100
                idf = self.vectorizer.idf_[idx]
                importance_score = importance * idf
                
                self.term_weights[term] = importance_score
                
                if freq_pct >= self.config.common_term_threshold:
                    self.common_terms.add(term)
            
            # Create pattern for common terms

            self.common_terms_pattern = _build_term_pattern(
                self.common_terms
            )
            
            self.fitted = True
            self.similarity_cache.clear()
            
            # Log significant terms
            self._log_significant_terms(term_doc_freq, n_docs)
            
        except Exception as e:
            logging.error(f"Error in TF-IDF fit: {e}")
            self.fitted = False
    
    def _log_significant_terms(self, term_doc_freq: np.ndarray, n_docs: int) -> None:
        """Log most significant terms for analysis."""
        significant_terms = sorted(
            self.term_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        logging.info("\nMost significant terms by TF-IDF weight:")
        for term, weight in significant_terms:
            freq = term_doc_freq[self.vectorizer.vocabulary_[term]]
            freq_pct = (freq / n_docs) * 100
            logging.info(
                f"- {term:<20}: weight={weight:.3f}, freq={freq_pct:.1f}%"
            )
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not self.fitted:
            return 0.0
            
        cache_key = (text1.lower(), text2.lower())
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        try:
            vec1 = self.vectorizer.transform([text1.lower()])
            vec2 = self.vectorizer.transform([text2.lower()])
            
            similarity = self._calculate_similarity(vec1, vec2)
            
            if len(self.similarity_cache) < self.MAX_CACHE_SIZE:
                self.similarity_cache[cache_key] = similarity
                
            return similarity
            
        except Exception as e:
            logging.warning(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def _calculate_similarity(self, vec1: csr_matrix, vec2: csr_matrix) -> float:
        """Calculate weighted cosine similarity between vectors."""
        weighted_vec1 = vec1.copy()
        weighted_vec2 = vec2.copy()
        
        # Apply term weights
        for term, idx in self.vectorizer.vocabulary_.items():
            weight = self.term_weights.get(term, 1.0)
            weighted_vec1.data[weighted_vec1.indices == idx] *= weight
            weighted_vec2.data[weighted_vec2.indices == idx] *= weight
        
        # Calculate similarity
        num = weighted_vec1.multiply(weighted_vec2).sum()
        norm1 = np.sqrt(weighted_vec1.multiply(weighted_vec1).sum())
        norm2 = np.sqrt(weighted_vec2.multiply(weighted_vec2).sum())
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
            
        return float(num / (norm1 * norm2))