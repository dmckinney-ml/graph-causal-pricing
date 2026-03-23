"""
Temporal train / validation / test splits for the causal panel.

Causal estimation must respect time ordering — no random shuffling.
Split boundaries are defined in configs/causal_config.yaml.
"""

from __future__ import annotations

import pandas as pd


def create_temporal_split(
    panel: pd.DataFrame,
    train_weeks: tuple[int, int] = (1, 70),
    val_weeks: tuple[int, int] = (71, 85),
    test_weeks: tuple[int, int] = (86, 102),
) -> dict[str, pd.DataFrame]:
    """
    Split the panel by WEEK_NO into train / val / test.

    Parameters
    ----------
    panel       : DataFrame with a ``WEEK_NO`` column.
    train_weeks : (min_week, max_week) inclusive for training.
    val_weeks   : (min_week, max_week) inclusive for validation.
    test_weeks  : (min_week, max_week) inclusive for testing.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` → DataFrames.
    """
    return {
        "train": panel[
            panel["WEEK_NO"].between(train_weeks[0], train_weeks[1])
        ].reset_index(drop=True),
        "val": panel[
            panel["WEEK_NO"].between(val_weeks[0], val_weeks[1])
        ].reset_index(drop=True),
        "test": panel[
            panel["WEEK_NO"].between(test_weeks[0], test_weeks[1])
        ].reset_index(drop=True),
    }
