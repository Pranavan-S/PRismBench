from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional


DEFAULT_LABEL_COLUMNS: List[str] = [
    "bug",
    "security",
    "performance",
    "code_quality_or_maintenability",
    "non_risky",
]

DEFAULT_CATEG_COLS: List[str] = [
    "linked_issue_priority",
    "linked_issue_resolution",
    "linked_issue_status",
    "linked_issue_type",
]

DEFAULT_DROP_COLS: List[str] = DEFAULT_CATEG_COLS + [
    "linked_issue_subtasks_count",
    "linked_issue_has_data_loss_keywords",
    "linked_issue_has_error_keywords",
    "linked_issue_has_exception_keywords",
    "linked_issue_has_performance_keywords",
    "linked_issue_has_security_keywords",
    "linked_issue_has_stack_trace",
    "pr_number",
    "historical_features_computed",
]

DEFAULT_NUMERIC_COLS: List[str] = [
    "additions",
    "author_account_age_days",
    "author_avg_pr_size",
    "author_followers",
    "author_following",
    "author_public_repos",
    "author_total_buggy_prs",
    "author_total_gists",
    "author_total_prs_in_project",
    "author_total_repos",
    "deletions",
    "description_length",
    "description_unique_words",
    "description_word_count",
    "files_added",
    "files_deleted",
    "files_modified",
    "files_renamed",
    "linked_issue_age_days",
    "linked_issue_comments_count",
    "linked_issue_components_count",
    "linked_issue_description_length",
    "linked_issue_description_unique_words",
    "linked_issue_description_word_count",
    "linked_issue_readability_cli",
    "linked_issue_severity_score",
    "linked_issue_technical_keyword_count",
    "linked_issue_title_length",
    "linked_issue_title_unique_words",
    "linked_issue_title_word_count",
    "linked_issue_total_interactions",
    "num_commits",
    "pr_lifespan_days",
    "readability_cli",
    "src_churn",
    "src_files_count",
    "technical_keyword_count",
    "test_churn",
    "test_files_count",
    "test_to_src_ratio",
    "title_length",
    "title_unique_words",
    "title_word_count",
    "total_churn",
]


@dataclass(frozen=True)
class SelectionConfig:
    """
    Configuration for uncertainty selection.

    Notes:
      - This mirrors the notebook's defaults. Adjust as needed.
      - The script assumes the input CSVs contain:
          * 'pr_number' in the unlabeled CSV (for mapping scores to PR IDs)
          * the label columns in labeled train/test CSVs
    """

    loop_number: int = 1
    project_name: str = "kafka"

    data_root: Path = Path("SamplingLoopData")
    output_dir: Path = Path("UncertainPoints")

    label_columns: Sequence[str] = tuple(DEFAULT_LABEL_COLUMNS)
    drop_columns: Sequence[str] = tuple(DEFAULT_DROP_COLS)
    numeric_columns: Sequence[str] = tuple(DEFAULT_NUMERIC_COLS)

    # Bootstrap ensemble
    n_bootstrap_sets: int = 20
    random_state: int = 42

    # Selection
    top_uncertain_pool: int = 100  # choose top-N uncertain, then diversify
    k_diverse: int = 25            # k-center greedy within the top_uncertain_pool
    top_k_labels_for_score: int = 3  # sum of top-k label entropies across label per PR

    # Distance metric for k-center greedy
    metric: str = "euclidean"
