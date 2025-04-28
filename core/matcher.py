"""Main record matching system implementation."""

from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import logging
from functools import partial
import time
from datasketch import MinHash, MinHashLSH
from pybloom_live import ScalableBloomFilter
import traceback
from collections import defaultdict
import xxhash

from core.preprocessor import PreprocessorRegistry
from core.validator import StringValidator
from core.analyzer import TFIDFAnalyzer
from config.models import (
    MatchStrategy, 
    ColumnMatchConfig,
    PartialMatchConfig,
    PartialMatchResult
)
from config.rules import AdditionRules

class RecordMatcher:
    """
    Main record matching system implementing multiple matching strategies.
    """
    
    def __init__(
        self,
        strategies: List[MatchStrategy],
        addition_rules: AdditionRules,
        minhash_permutations: int = 256,
        worker_processes: int = -1,
        cache_size: int = 10000,
        include_match_details: bool = False,
        enforce_validation: bool = False
    ):
        """
        Initialize the record matcher.
        
        Args:
            strategies: List of matching strategies to use
            addition_rules: Rules for adding columns from matches
            minhash_permutations: Number of permutations for MinHash
            worker_processes: Number of worker processes (-1 for CPU count)
            cache_size: Size of various caches
            include_match_details: Whether to include detailed match information
            enforce_validation: Whether to enforce partial validation
        """
        self.strategies = strategies
        self.addition_rules = addition_rules
        self.minhash_permutations = minhash_permutations
        self.worker_processes = worker_processes if worker_processes > 0 else cpu_count()
        self.cache_size = cache_size
        self.include_match_details = include_match_details
        self.enforce_validation = enforce_validation
        
        # Initialize components
        self.preprocessor_registry = PreprocessorRegistry()
        self.string_validator = StringValidator()
        self.tfidf_analyzers: Dict[str, TFIDFAnalyzer] = {}
        
        # Initialize indexes
        self.lsh_indexes: Dict[str, MinHashLSH] = {}
        self.bloom_filters: Dict[str, ScalableBloomFilter] = {}
        self.inverted_indexes: Dict[str, Dict[bytes, Set[int]]] = {}
        
        self._initialize_logging()
    
    def _initialize_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_analyzers(self, df: pd.DataFrame) -> None:
        """Initialize TF-IDF analyzers for relevant columns."""
        for strategy in self.strategies:
            for col_config in strategy.column_configs:
                if (col_config.tfidf_config and 
                    col_config.tfidf_config.enabled and 
                    col_config.name in df.columns):
                    
                    if col_config.name not in self.tfidf_analyzers:
                        self.tfidf_analyzers[col_config.name] = TFIDFAnalyzer(
                            config=col_config.tfidf_config,
                            preprocessor=self.preprocessor_registry.create(
                                col_config.preprocess_method
                            ).process
                        )
                    
                    self.tfidf_analyzers[col_config.name].fit(
                        df[col_config.name].fillna('').astype(str).tolist()
                    )
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess all relevant columns in parallel.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        with ThreadPoolExecutor(max_workers=self.worker_processes) as executor:
            futures = []
            
            for strategy in self.strategies:
                for col_config in strategy.column_configs:
                    if col_config.name in df.columns:
                        # Create preprocessor with appropriate config
                        config = {}
                        if col_config.date_config:
                            config['date_config'] = col_config.date_config
                            
                        preprocessor = self.preprocessor_registry.create(
                            col_config.preprocess_method,
                            **config
                        )
                        
                        # Create a function that closes over the correct preprocessor instance
                        def make_processor_func(processor):
                            return lambda x: x.apply(processor.process)
                        
                        processor_func = make_processor_func(preprocessor)
                        
                        if col_config.preprocess_method == 'postal_code':
                            print("CORRECT")
                        print(col_config.name, col_config.preprocess_method)
                        
                        future = executor.submit(
                            processor_func,
                            df[col_config.name]
                        )
                        futures.append((
                            f'{col_config.name}_processed',
                            future
                        ))
            
            for col_name, future in futures:
                processed_df[col_name] = future.result()
                if "post" in col_name:
                    print(col_name, processed_df[col_name].head())
        
        return processed_df


    def _create_minhash(self, values: Tuple[str, ...]) -> MinHash:
        """Create MinHash signature for a set of values."""
        minhash = MinHash(num_perm=self.minhash_permutations)
        for value in values:
            if value:
                minhash.update(xxhash.xxh64(value).digest())
        return minhash

    def _build_indexes(self, df: pd.DataFrame) -> None:
        """Build all necessary indexes for the reference dataset."""
        for strategy in self.strategies:
            if not all(col_config.name in df.columns 
                      for col_config in strategy.column_configs):
                continue

            lsh = MinHashLSH(
                threshold=strategy.min_threshold,
                num_perm=self.minhash_permutations
            )
            bloom_filter = ScalableBloomFilter(
                mode=ScalableBloomFilter.LARGE_SET_GROWTH
            )
            inverted_index = defaultdict(set)
            
            processed_columns = [
                f'{col_config.name}_processed' 
                for col_config in strategy.column_configs 
                if col_config.name in df.columns
            ]
            
            for idx, row in df[processed_columns].iterrows():
                values = tuple(row)
                
                # Add to LSH index
                minhash = self._create_minhash(values)
                lsh.insert(idx, minhash)
                
                # Add to Bloom filter
                bloom_filter.add(idx)
                
                # Add to inverted index
                for col_name, value in zip(processed_columns, values):
                    if value:
                        key = xxhash.xxh64(f"{col_name}:{value}").digest()
                        inverted_index[key].add(idx)
            
            self.lsh_indexes[strategy.name] = lsh
            self.bloom_filters[strategy.name] = bloom_filter
            self.inverted_indexes[strategy.name] = inverted_index
    
    def _find_candidates(
        self,
        row: pd.Series,
        strategy: MatchStrategy
    ) -> Set[int]:
        """
        Find candidate matches for a row using all available indexes.
        
        Args:
            row: Row to find candidates for
            strategy: Matching strategy to use
            
        Returns:
            Set[int]: Set of candidate indices
        """
        values = tuple(
            row[f'{col_config.name}_processed'] 
            for col_config in strategy.column_configs 
            if f'{col_config.name}_processed' in row.index
        )
        
        # Get candidates from LSH
        minhash = self._create_minhash(values)
        candidates = set(self.lsh_indexes[strategy.name].query(minhash))
        
        # Get candidates from inverted index
        for col_name, value in zip(
            [f'{col_config.name}_processed' for col_config in strategy.column_configs],
            values
        ):
            if value:
                key = xxhash.xxh64(f"{col_name}:{value}").digest()
                candidates.update(self.inverted_indexes[strategy.name][key])
        
        # Filter through Bloom filter
        return {c for c in candidates if self.bloom_filters[strategy.name].add(c)}

    def _calculate_match_score(
        self,
        values1: Tuple[str, ...],
        values2: Tuple[str, ...],
        strategy: MatchStrategy
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate similarity score between two records.
        
        Args:
            values1: First record's values
            values2: Second record's values
            strategy: Matching strategy to use
            
        Returns:
            Tuple[float, bool, Dict]: Score, whether it's a desired match, and details
        """
        total_weight = sum(col_config.weight for col_config in strategy.column_configs)
        weighted_similarity = 0
        is_desired_match = True
        validation_results = {}

        for col_config, val1, val2 in zip(strategy.column_configs, values1, values2):
            if col_config.is_required and (not val1 or not val2):
                return 0, False, {}

            if val1 or val2:
                # Calculate base similarity
                if col_config.match_method == 'exact':
                    similarity = 1.0 if val1.lower() == val2.lower() else 0.0
                else:
                    analyzer = self.tfidf_analyzers.get(col_config.name)
                    if analyzer and analyzer.common_terms_pattern:
                        val1 = analyzer.common_terms_pattern.sub('', val1).strip()
                        val2 = analyzer.common_terms_pattern.sub('', val2).strip()

                    
                    similarity = self.string_validator.calculate_similarity(
                        val1,
                        val2
                    )

                # Try partial matching if needed
                if (similarity < col_config.desired_threshold and 
                    col_config.partial_match_config):
                    
                    partial_score, partial_results = self.string_validator.validate_partial_matches(
                        val1,
                        val2,
                        col_config.partial_match_config
                    )
                    
                    if partial_score >= col_config.partial_match_config.required_match_percentage:
                        similarity = max(similarity, partial_score)
                        validation_results[col_config.name] = {
                            'is_validated': True,
                            'partial_matches': partial_results
                        }
                    else:
                        validation_results[col_config.name] = {
                            'is_validated': False,
                            'partial_matches': partial_results
                        }
                else:
                    validation_results[col_config.name] = {
                        'is_validated': False,
                        'partial_matches': []
                    }

                # Check thresholds
                threshold = (
                    col_config.strict_threshold or col_config.min_threshold
                    if len(str(val1)) < col_config.min_content_length or 
                       len(str(val2)) < col_config.min_content_length
                    else col_config.min_threshold
                )

                if similarity >= threshold:
                    weighted_similarity += similarity * col_config.weight
                    if similarity < col_config.desired_threshold:
                        is_desired_match = False
                else:
                    return 0, False, {}
            else:
                weighted_similarity += col_config.weight
                validation_results[col_config.name] = {
                    'is_validated': False,
                    'partial_matches': []
                }

        final_similarity = weighted_similarity / total_weight if total_weight > 0 else 0
        return (
            final_similarity,
            is_desired_match and final_similarity >= strategy.desired_threshold,
            validation_results
        )

    def match_dataframes(
        self,
        source_df: pd.DataFrame,
        target_df: Optional[pd.DataFrame] = None,
        match_filename: str = 'match'
    ) -> pd.DataFrame:
        """
        Match records between two dataframes or deduplicate a single dataframe.
        
        Args:
            source_df: Source DataFrame
            target_df: Optional target DataFrame
            match_filename: Name for match file
            
        Returns:
            pd.DataFrame: DataFrame with match results
        """
        start_time = time.time()
        
        # Validate and prepare strategies
        available_strategies = self._validate_strategies(source_df, target_df)
        if not available_strategies:
            self.logger.warning(
                "No valid strategies found. Returning unmatched data."
            )
            result_df = source_df.copy()
            self._add_default_match_data(result_df)
            return result_df

        # Preprocess data
        source_df_processed = self._preprocess_dataframe(source_df)
        operation = 'deduplicate' if target_df is None else 'link'
        
        if target_df is not None:
            target_df_processed = self._preprocess_dataframe(target_df)
            self._initialize_analyzers(pd.concat([source_df, target_df]))
            self._build_indexes(target_df_processed)
        else:
            self._initialize_analyzers(source_df)
            self._build_indexes(source_df_processed)
            target_df_processed = source_df_processed

        # Process in parallel
        chunks = np.array_split(source_df_processed, self.worker_processes)
        
        with Pool(processes=self.worker_processes) as pool:
            results = pool.map(
                partial(
                    self._process_chunk,
                    target_df=target_df_processed,
                    operation=operation,
                    match_filename=match_filename
                ),
                chunks
            )

        # Combine results
        result_df = pd.DataFrame([
            item for sublist in results for item in sublist
        ])
        
        if not self.include_match_details:
            columns_to_drop = [
                col for col in result_df.columns 
                if col.endswith('_processed')
            ]
            result_df = result_df.drop(columns=columns_to_drop)

        self.logger.info(
            f"Matching completed in {time.time() - start_time:.2f} seconds"
        )
        
        return result_df
    
    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        target_df: pd.DataFrame,
        operation: str,
        match_filename: str
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of records for matching.
        
        Args:
            chunk: DataFrame chunk to process
            target_df: Target DataFrame to match against
            operation: Type of operation ('link' or 'deduplicate')
            match_filename: Name for match file
        
        Returns:
            List[Dict[str, Any]]: List of match results
        """
        results = []
        
        for _, row in chunk.iterrows():
            best_match = None
            best_similarity = 0
            best_strategy = None
            is_desired_match = False
            best_validation_results = {}
            
            for strategy in self.strategies:
                if not self._check_required_columns(row, strategy):
                    continue

                # Find candidate matches
                candidates = self._find_candidates(row, strategy)
                values1 = self._get_processed_values(row, strategy)
                
                # Check each candidate
                for candidate_idx in candidates:
                    candidate_row = target_df.loc[candidate_idx]
                    values2 = self._get_processed_values(candidate_row, strategy)
                    
                    try:
                        similarity, desired, validation_results = self._calculate_match_score(
                            values1, values2, strategy
                        )
                        
                        accept_match = True
                        if self.enforce_validation and strategy.require_partial_validation:
                            if similarity >= strategy.min_threshold and similarity < strategy.desired_threshold:
                                parts_validated = self._check_validation(validation_results, strategy)
                                accept_match = parts_validated

                        if accept_match and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = candidate_row
                            best_strategy = strategy
                            is_desired_match = desired
                            best_validation_results = validation_results
                    except Exception as e:
                        self.logger.warning(f"Error calculating similarity: {traceback.format_exc()}")
                        continue

            # Create result record
            result = self._create_result_record(
                row, best_match, best_similarity, best_strategy,
                is_desired_match, best_validation_results, operation,
                match_filename
            )
            results.append(result)

        return results

    def _check_required_columns(self, row: pd.Series, strategy: MatchStrategy) -> bool:
        """Check if all required columns exist in row."""
        return all(
            f'{col_config.name}_processed' in row.index
            for col_config in strategy.column_configs
            if col_config.is_required
        )

    def _get_processed_values(
        self,
        row: pd.Series,
        strategy: MatchStrategy
    ) -> Tuple[str, ...]:
        """Get processed values for columns in strategy."""
        return tuple(
            row[f'{col_config.name}_processed']
            if f'{col_config.name}_processed' in row.index else ''
            for col_config in strategy.column_configs
        )

    def _check_validation(
        self,
        validation_results: Dict[str, Any],
        strategy: MatchStrategy
    ) -> bool:
        """Check if validation requirements are met."""
        return any(
            validation_results.get(col_config.name, {}).get('is_validated', False)
            for col_config in strategy.column_configs
            if col_config.is_required and col_config.partial_match_config
        )

    def _create_result_record(
        self,
        row: pd.Series,
        match: Optional[pd.Series],
        similarity: float,
        strategy: Optional[MatchStrategy],
        is_desired: bool,
        validation_results: Dict[str, Any],
        operation: str,
        match_filename: str
    ) -> Dict[str, Any]:
        """Create a result record with match information."""
        result = {
            col: str(val) if pd.notna(val) else ''
            for col, val in row.items()
        }

        if match is not None and similarity >= strategy.min_threshold:
            self._add_match_data(
                result, match, similarity, strategy,
                is_desired, validation_results, operation,
                match_filename
            )
        else:
            self._add_default_match_data(result)

        return result

    def _add_match_data(
        self,
        result: Dict[str, Any],
        match: pd.Series,
        similarity: float,
        strategy: MatchStrategy,
        is_desired: bool,
        validation_results: Dict[str, Any],
        operation: str,
        match_filename: str
    ) -> None:
        """Add match-related data to result record."""
        # Add basic match info
        result.update({
            'is_matched': True,
            'match_similarity': similarity,
            'matching_strategy': strategy.name,
            'matched_record_id': match.name,
            'is_desired_match': is_desired
        })

        # Add validation details if requested
        if self.include_match_details:
            for col_name, validation in validation_results.items():
                result[f'{col_name}_validated'] = validation.get('is_validated', False)
                if validation.get('partial_matches'):
                    result[f'{col_name}_match_details'] = [
                        self._format_match_detail(m)
                        for m in validation.get('partial_matches', [])
                    ]

        # Add matched columns according to rules
        if operation == 'link':
            self._process_match_columns(match, result, match_filename)

    def _add_default_match_data(self, result: Dict[str, Any]) -> None:
        """Add default data for unmatched record."""
        result.update({
            'is_matched': False,
            'match_similarity': 0,
            'matching_strategy': None,
            'matched_record_id': None,
            'is_desired_match': False
        })

        if self.include_match_details:
            for col_config in self.strategies[0].column_configs:
                result[f'{col_config.name}_validated'] = False
                result[f'{col_config.name}_match_details'] = None

    def _process_match_columns(
        self,
        match: pd.Series,
        base_record: Dict[str, Any],
        match_filename: str
    ) -> None:
        """
        Process match columns to both merge data and add new columns based on rules.
        
        This function handles both:
        1. Merging data from match record where base record is empty
        2. Adding new columns based on addition rules
        
        Args:
            match: Matched record from target dataset
            base_record: Base record to update
            match_filename: Name of match file for special column naming
        """
        # Skip these system columns
        system_columns = {
            'is_matched', 'match_similarity', 'matching_strategy',
            'matched_record_id', 'is_desired_match'
        }

        existing_columns = set(base_record.keys())
        
        for col_name, match_value in match.items():
            # Skip processed and system columns
            if col_name.endswith('_processed') or col_name in system_columns:
                continue
                
            # Convert match value to string if not null
            match_value_str = str(match_value) if pd.notna(match_value) else ''
            
            # Case 1: Column exists in base record
            if col_name in existing_columns:
                base_value = base_record[col_name]
                # Update if base value is empty
                if pd.isna(base_value) or str(base_value).strip() == '':
                    base_record[col_name] = match_value_str
            
            # Case 2: Column should be added based on rules
            elif self.addition_rules.should_add_column(col_name, list(existing_columns)):
                # Handle special column naming for hash columns
                new_col_name = col_name
                base_record[new_col_name] = match_value_str

    def _format_match_detail(self, match_result: PartialMatchResult) -> Dict[str, Any]:
        """Format partial match result for output."""
        return {
            'source_part': match_result.source_part,
            'target_part': match_result.target_part,
            'similarity': match_result.similarity_score,
            'match_type': match_result.match_category.value,
            'points': match_result.score_contribution,
            'position': match_result.position_index,
            'spaced_match': match_result.spaced_match,
            'compressed_match': match_result.compressed_match
        }

    def _validate_strategies(
        self,
        source_df: pd.DataFrame,
        target_df: Optional[pd.DataFrame]
    ) -> List[MatchStrategy]:
        """Validate and return available strategies."""
        available_strategies = []
        
        for strategy in self.strategies:
            columns_exist = True
            for col_config in strategy.column_configs:
                if col_config.is_required:
                    if col_config.name not in source_df.columns:
                        self.logger.warning(
                            f"Required column {col_config.name} not found in source "
                            f"dataframe for strategy {strategy.name}. Skipping strategy."
                        )
                        columns_exist = False
                        break
                    if target_df is not None and col_config.name not in target_df.columns:
                        self.logger.warning(
                            f"Required column {col_config.name} not found in target "
                            f"dataframe for strategy {strategy.name}. Skipping strategy."
                        )
                        columns_exist = False
                        break
            
            if columns_exist:
                available_strategies.append(strategy)
        
        return available_strategies