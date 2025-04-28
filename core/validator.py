"""String similarity and validation functionality."""

import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import re
import Levenshtein
from functools import lru_cache
from config.models import PartialMatchConfig, MatchCategory, PartialMatchResult


class StringValidator:
    """Validates and compares strings with configurable matching strategies."""
    
    # Similarity thresholds
    EXACT_THRESHOLD = 0.99
    HIGH_SIMILARITY_THRESHOLD = 0.90
    MEDIUM_SIMILARITY_THRESHOLD = 0.80
    
    # Score multipliers
    EXACT_MATCH_MULTIPLIER = 1.0
    HIGH_SIMILARITY_MULTIPLIER = 0.9
    MEDIUM_SIMILARITY_MULTIPLIER = 0.7
    
    def __init__(self):
        """Initialize validator with caching."""
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
    
    @staticmethod
    def _calculate_base_points(word: str) -> float:
        """
        Calculate base points for a word based on length and complexity.
        
        Args:
            word: Word to calculate points for
            
        Returns:
            float: Base points for the word
        """
        length = len(word)
        
        # Basic points based on length
        if length <= 2:
            base_points = 0.5
        elif length <= 3:
            base_points = 1.0
        elif length <= 5:
            base_points = 2.0
        elif length <= 8:
            base_points = 3.0
        else:
            base_points = 4.0
        
        # Bonus for word complexity
        unique_chars = len(set(word))
        complexity_ratio = unique_chars / length
        complexity_bonus = base_points * (complexity_ratio - 0.5)
        
        return base_points + max(0, complexity_bonus)
    
    def _calculate_position_bonus(
        self,
        pos1: int,
        pos2: int,
        total_len1: int,
        total_len2: int
    ) -> float:
        """Calculate bonus based on word positions."""
        # Perfect position match
        if pos1 == pos2:
            if pos1 == 0:  # Start of string
                return 0.3
            elif pos1 == total_len1 - 1 and pos2 == total_len2 - 1:  # End
                return 0.2
            else:  # Middle positions
                return 0.15
        
        # Near position match
        pos_diff = abs(pos1 - pos2)
        max_positions = max(total_len1, total_len2)
        
        # Scale penalty based on string lengths
        if max_positions <= 2:
            position_penalty = 0.4 * pos_diff
        elif max_positions <= 4:
            position_penalty = 0.3 * pos_diff
        else:
            position_penalty = 0.2 * pos_diff
        
        return max(0, 0.2 - position_penalty)
    
    @lru_cache(maxsize=10000)
    def _calculate_word_similarity(
        self,
        word1: str,
        word2: str,
        compressed: bool = False
    ) -> float:
        """Calculate cached similarity between two words."""
        if not word1 or not word2:
            return 0.0
            
        if word1 == word2:
            return 1.0
            
        # For very short words, require exact match
        if len(word1) <= 3 or len(word2) <= 3:
            return 1.0 if word1 == word2 else 0.0
        
        # Handle compressed comparison differently
        if compressed:
            w1 = ''.join(word1.split())
            w2 = ''.join(word2.split())
            if w1 == w2:
                return 1.0
        
        return 1 - (Levenshtein.distance(word1, word2) / max(len(word1), len(word2)))
    
    def calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate overall similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not isinstance(s1, str) or not isinstance(s2, str):
            return 0.0

        try:
            # Safely split the strings, providing defaults if no separator
            s1_parts = s1.split('|', 1) if '|' in s1 else [s1, ''.join(s1.split())]
            s2_parts = s2.split('|', 1) if '|' in s2 else [s2, ''.join(s2.split())]
            
            s1_spaced, s1_compressed = s1_parts
            s2_spaced, s2_compressed = s2_parts
            
            # Quick exact match checks
            if s1_compressed == s2_compressed:
                return 1.0
            
            # Calculate similarities for both versions
            spaced_similarity = self._word_based_similarity(
                s1_spaced.split(),
                s2_spaced.split(),
                False
            )
            compressed_similarity = self._word_based_similarity(
                [s1_compressed],
                [s2_compressed],
                True
            )
            
            return max(spaced_similarity, compressed_similarity)
            
        except Exception as e:
            logging.warning(f"Error in calculate_similarity: {str(e)}")
            # Fallback to basic comparison if the enhanced comparison fails
            return self._basic_similarity(s1, s2)
    
    def _word_based_similarity(
        self,
        words1: List[str],
        words2: List[str],
        compressed: bool
    ) -> float:
        """Calculate similarity based on word matching."""
        if not words1 or not words2:
            return 0.0
        
        matched_words = 0
        used_indices = set()
        total_similarity = 0.0
        
        for i, w1 in enumerate(words1):
            best_match = 0.0
            best_idx = -1
            
            for j, w2 in enumerate(words2):
                if j not in used_indices:
                    similarity = self._calculate_word_similarity(w1, w2, compressed)
                    
                    if similarity > best_match:
                        best_match = similarity
                        best_idx = j
            
            if best_match >= self.MEDIUM_SIMILARITY_THRESHOLD:
                matched_words += 1
                total_similarity += best_match
                if best_idx != -1:
                    used_indices.add(best_idx)
        
        # Calculate components
        word_match_ratio = matched_words / max(len(words1), len(words2))
        avg_similarity = total_similarity / len(words1) if matched_words > 0 else 0
        len_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
        
        return (
            0.6 * word_match_ratio +  # Weight of matched count
            0.3 * avg_similarity +    # Weight of match quality
            0.1 * len_ratio          # Weight of length similarity
        )
    
    def validate_partial_matches(
        self,
        text1: str,
        text2: str,
        config: PartialMatchConfig
    ) -> Tuple[float, List[PartialMatchResult]]:
        """
        Validate partial matches between strings.
        
        Args:
            text1: First string
            text2: Second string
            config: Partial matching configuration
            
        Returns:
            Tuple[float, List[PartialMatchResult]]: Overall score and detailed matches
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0, []
        
        if len(text1) < config.min_length or len(text2) < config.min_length:
            return 0.0, []

        try:
            # Safely split the strings, providing defaults if no separator
            text1_parts = text1.split('|', 1) if '|' in text1 else [text1, ''.join(text1.split())]
            text2_parts = text2.split('|', 1) if '|' in text2 else [text2, ''.join(text2.split())]
            
            text1_spaced, text1_compressed = text1_parts
            text2_spaced, text2_compressed = text2_parts
            
            # Get word lists from spaced versions
            words1 = text1_spaced.split()
            words2 = text2_spaced.split()
            
            if not words1 or not words2:
                return 0.0, []
            
            # Calculate maximum possible points
            total_max_points = sum(self._calculate_base_points(w) for w in words1)
            
            # Check length ratio
            length_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
            if length_ratio < 0.3:  # Too different in length
                return 0.0, []
            
            # Match words and create results
            matches = self._find_matching_parts(
                words1, words2,
                text1_compressed, text2_compressed
            )
            
            if not matches:
                return 0.0, []
            
            # Calculate final score
            total_points = sum(m.score_contribution for m in matches)
            final_score = total_points / total_max_points if total_max_points > 0 else 0.0
            
            # Apply adjustments
            final_score = self._adjust_score(final_score, matches, words1, words2)
            
            # Apply threshold
            min_threshold = 0.7
            if len(words1) <= 2 or len(words2) <= 2:
                min_threshold = 0.85
            elif len(words1) <= 3 or len(words2) <= 3:
                min_threshold = 0.8
            
            if final_score < min_threshold:
                return 0.0, []
            
            return final_score, matches
            
        except Exception as e:
            logging.warning(f"Error in validate_partial_matches: {str(e)}")
            return 0.0, []

    
    def _find_matching_parts(
        self,
        words1: List[str],
        words2: List[str],
        compressed1: str,
        compressed2: str
    ) -> List[PartialMatchResult]:
        """Find matching parts between strings."""
        matches = []
        used_positions = set()
        
        for i, word1 in enumerate(words1):
            best_match = None
            best_pos = -1
            best_similarity = 0
            best_points = 0
            best_spaced = False
            best_compressed = False
            
            for j, word2 in enumerate(words2):
                if j not in used_positions:
                    # Try both spaced and compressed matching
                    spaced_sim = self._calculate_word_similarity(word1, word2, False)
                    compressed_sim = self._calculate_word_similarity(
                        ''.join(word1.split()),
                        ''.join(word2.split()),
                        True
                    )
                    
                    similarity = max(spaced_sim, compressed_sim)
                    if similarity >= self.MEDIUM_SIMILARITY_THRESHOLD:
                        points = (
                            self._calculate_base_points(word1) *
                            self._get_multiplier(similarity) *
                            (1 + self._calculate_position_bonus(i, j, len(words1), len(words2)))
                        )
                        
                        if points > best_points:
                            best_match = word2
                            best_pos = j
                            best_similarity = similarity
                            best_points = points
                            best_spaced = spaced_sim >= self.MEDIUM_SIMILARITY_THRESHOLD
                            best_compressed = compressed_sim >= self.MEDIUM_SIMILARITY_THRESHOLD
            
            if best_match and best_points > 0:
                matches.append(PartialMatchResult(
                    source_part=word1,
                    target_part=best_match,
                    similarity_score=best_similarity,
                    match_category=self._get_match_category(best_similarity),
                    score_contribution=best_points,
                    position_index=(i, best_pos),
                    spaced_match=best_spaced,
                    compressed_match=best_compressed
                ))
                used_positions.add(best_pos)
        
        return matches
    
    def _get_multiplier(self, similarity: float) -> float:
        """Get score multiplier based on similarity level."""
        if similarity >= self.EXACT_THRESHOLD:
            return self.EXACT_MATCH_MULTIPLIER
        elif similarity >= self.HIGH_SIMILARITY_THRESHOLD:
            return self.HIGH_SIMILARITY_MULTIPLIER
        else:
            return self.MEDIUM_SIMILARITY_MULTIPLIER
    
    def _get_match_category(self, similarity: float) -> MatchCategory:
        """Get match category based on similarity level."""
        if similarity >= self.EXACT_THRESHOLD:
            return MatchCategory.EXACT
        elif similarity >= self.MEDIUM_SIMILARITY_THRESHOLD:
            return MatchCategory.SIMILAR
        return MatchCategory.NO_MATCH
    
    def _adjust_score(
        self,
        score: float,
        matches: List[PartialMatchResult],
        words1: List[str],
        words2: List[str]
    ) -> float:
        """Apply contextual adjustments to the score."""
        # Adjust for match ratio
        matched_ratio = len(matches) / max(len(words1), len(words2))
        if matched_ratio < 0.5:
            score *= matched_ratio * 1.5
        
        # Penalize if first word doesn't match well
        first_word_matched = any(
            m for m in matches 
            if m.position_index == (0, 0) and m.similarity_score >= 0.9
        )
        if not first_word_matched:
            score *= 0.8
        
        # Penalize sequence breaks
        if len(matches) > 1:
            sequence_breaks = sum(
                1 for i in range(len(matches) - 1)
                if matches[i].position_index[1] > matches[i + 1].position_index[1]
            )
            if sequence_breaks > 0:
                score *= max(0.7, 1 - (sequence_breaks * 0.15))
        
        return score