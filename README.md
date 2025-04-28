# Record Matcher

A flexible and efficient system for matching records across datasets using multiple strategies, including TF-IDF analysis, fuzzy matching, and configurable validation rules.

## Features

* Multiple matching strategies with configurable thresholds and weights.
* Advanced text matching using TF-IDF analysis to identify important terms.
* Flexible preprocessing pipeline for standardizing data, including handling names, postal codes, and dates.
* Support for both linking (matching records between two datasets) and deduplication (finding duplicates within a single dataset).
* Parallel processing to improve performance on large datasets.
* Detailed match analysis and reporting, including match similarity, strategy used, and validation details.
* Configurable rules for adding columns from matched records.

## Installation

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Install the required dependencies. It's recommended to use a virtual environment.

    ```bash
    pip install pandas numpy scikit-learn datasketch pybloom_live xxhash python-Levenshtein python-dateutil regex xlsxwriter
    ```

## Usage

1.  Define your matching strategies, column configurations, and addition rules using the models in `config/models.py` and `config/rules.py`. Refer to `config/matcher_example.py` for an example of how to set up these configurations.
2.  Instantiate the `RecordMatcher` class with your defined strategies and rules.
3.  Load your source and (optionally) target data into pandas DataFrames.
4.  Use the `match_dataframes` method to perform the matching.

```python
import pandas as pd
from core.matcher import RecordMatcher
from config.models import MatchStrategy, ColumnMatchConfig, TFIDFConfig, PartialMatchConfig
from config.rules import AdditionRules, NewColumnsRule, PatternRule
from pathlib import Path

# Example Configuration (similar to matcher_example.py)
# ... (Define naam_match_config, strategies, addition_rules as in matcher_example.py) ...

# Create matcher instance
# matcher_instance = RecordMatcher(strategies=strategies, addition_rules=addition_rules, ...)

# Load dataframes
# df1 = pd.read_excel(Path('data/your_source_file.xlsx'), dtype=str).fillna('')
# df2 = pd.read_excel(Path('data/your_target_file.xlsx'), dtype=str).fillna('')

# Perform matching
# results_df = matcher_instance.match_dataframes(df1, df2)

# Print or save results
# print(results_df.head())
# results_df.to_excel(Path('data/matching_results.xlsx'), index=False)