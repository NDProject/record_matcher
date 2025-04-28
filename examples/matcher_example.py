"""Example usage of the record matching system with Excel files."""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from config.models import (
    MatchStrategy,
    ColumnMatchConfig,
    PartialMatchConfig,
    TFIDFConfig
)
from config.rules import (
    AdditionRules,
    NewColumnsRule,
    PatternRule
)
from core import matcher



def create_business_matcher(
    worker_processes: int = -1,
    include_match_details: bool = False,
    enforce_validation: bool = False
) -> matcher.RecordMatcher:
    """
    Create a matcher configured for business record matching.
    
    Args:
        worker_processes: Number of worker processes (-1 for CPU count)
        include_match_details: Whether to include detailed match information
        enforce_validation: Whether to enforce partial validation
    
    Returns:
        RecordMatcher: Configured matcher instance
    """
    # Create name matching configuration
    naam_match_config = ColumnMatchConfig(
        name='naam',
        weight=2.0,
        preprocess_method='name',
        match_method='fuzzy',
        min_threshold=0.75,
        desired_threshold=0.85,
        is_required=True,
        min_content_length=3,
        strict_threshold=0.9,
        partial_match_config=PartialMatchConfig(
            min_length=3,
            required_match_percentage=0.7
        ),
        tfidf_config=TFIDFConfig(
            enabled=True,
            min_df=0.01,
            max_df=0.9,
            ngram_range=(1, 3),
            min_term_length=4,
            common_term_threshold=3.0
        )
    )

    # Create matching strategies
    strategies = [
        MatchStrategy(
            "name_postcode",
            [
                naam_match_config,
                ColumnMatchConfig(
                    name='postcode',
                    weight=1.0,
                    preprocess_method='postal_code',
                    match_method='exact',
                    min_threshold=1.0,
                    is_required=True
                )
            ],
            min_threshold=0.7,
            desired_threshold=0.8,
            require_partial_validation=True
        ),
        MatchStrategy(
            "name_plaats",
            [
                naam_match_config,
                ColumnMatchConfig(
                    name='plaats',
                    weight=1.0,
                    preprocess_method='name',
                    match_method='exact',
                    min_threshold=1.0,
                    is_required=True
                )
            ],
            min_threshold=0.7,
            desired_threshold=0.8,
            require_partial_validation=True
        )
    ]

    # Create addition rules
    addition_rules = AdditionRules(
        include_rules=[
            NewColumnsRule(),  # Include new columns not in source
            PatternRule(r'url_.*'),  # Include URL columns
            PatternRule(r'#.*')  # Include hash columns
        ],
        exclude_columns=['internal_id', 'temp_column']  # Exclude specific columns
    )

    return matcher.RecordMatcher(
        strategies=strategies,
        addition_rules=addition_rules,
        minhash_permutations=128,
        worker_processes=worker_processes,
        include_match_details=include_match_details,
        enforce_validation=enforce_validation
    )

def match_excel_files(
    base_file: Path,
    match_file: Path,
    output_file: Optional[Path] = None,
    worker_processes: int = -1,
    include_match_details: bool = True
) -> pd.DataFrame:
    """
    Match records between two Excel files.
    
    Args:
        base_file: Path to base Excel file
        match_file: Path to match Excel file
        output_file: Optional path for output Excel file
        worker_processes: Number of worker processes
        include_match_details: Whether to include match details
        
    Returns:
        pd.DataFrame: DataFrame with match results
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create matcher
        matcher = create_business_matcher(
            worker_processes=worker_processes,
            include_match_details=include_match_details,
            enforce_validation=True
        )
        
        # Read Excel files
        logging.info(f"Reading base file: {base_file}")
        df1 = pd.read_excel(base_file, dtype=str).fillna('')
        
        logging.info(f"Reading match file: {match_file}")
        df2 = pd.read_excel(match_file, dtype=str).fillna('')
        
        # Perform matching
        logging.info("Starting matching process...")
        results = matcher.match_dataframes(df1, df2)
        
        # Log matching statistics
        total_records = len(results)
        matched_records = results['is_matched'].sum()
        desired_matches = results['is_desired_match'].sum()
        
        logging.info("\nMatching Statistics:")
        logging.info(f"Total records: {total_records}")
        logging.info(f"Matched records: {matched_records} ({matched_records/total_records*100:.1f}%)")
        logging.info(f"Desired matches: {desired_matches} ({desired_matches/total_records*100:.1f}%)")
        
        # Log validation statistics if available
        if 'naam_validated' in results.columns:
            logging.info("\nValidation Statistics:")
            validation_cols = [col for col in results.columns if col.endswith('_validated')]
            for col in validation_cols:
                valid_matches = results[results['is_matched']][col].sum()
                total_matches = results['is_matched'].sum()
                logging.info(
                    f"{col}: {valid_matches} validated out of {total_matches} matches "
                    f"({valid_matches/total_matches*100:.1f}%)"
                )
        
        # Log strategy effectiveness
        logging.info("\nStrategy Effectiveness:")
        strategy_counts = results[results['is_matched']]['matching_strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            avg_score = results[
                results['matching_strategy'] == strategy
            ]['match_similarity'].mean()
            logging.info(
                f"{strategy}: {count} matches, average score: {avg_score:.3f}"
            )
        
        # Save results if output file specified
        if output_file:
            logging.info(f"\nSaving results to: {output_file}")
            with pd.ExcelWriter(
                output_file,
                engine='xlsxwriter',
                engine_kwargs={'options': {'strings_to_urls': False}}
            ) as writer:
                results.to_excel(writer, index=False)
        
        return results
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage
    base_file = Path('data/base_data_.xlsx')
    match_file = Path('data/company_data.xlsx')
    output_file = Path('data/output_data.xlsx')
    
    results_df = match_excel_files(
        base_file=base_file,
        match_file=match_file,
        output_file=output_file,
        worker_processes=-1,  # Use all available CPU cores
        include_match_details=True
    )