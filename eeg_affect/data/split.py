"""Subject-independent cross-validation and train/val/test splitting strategies."""

from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


class SubjectGroupSplitter:
    """Splits data so that no participant appears in both training and test sets."""

    def __init__(
        self,
        testSize: float = 0.15,
        valSize: float = 0.10,
        randomState: int = 42,
        **kwargs,
    ):
        """Initializes the subject splitter.

        Args:
            testSize: Percentage of participants to hold out for testing (e.g. 15%).
            valSize: Percentage of participants to hold out for validation (e.g. 10%).
            randomState: Random seed so results are repeatable.
        """
        self.testSize = kwargs.get("test_size", testSize)
        self.test_size = self.testSize
        self.valSize = kwargs.get("val_size", valSize)
        self.val_size = self.valSize
        self.randomState = kwargs.get("random_state", randomState)
        self.random_state = self.randomState

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Splits the sample indices into Train, Validation, and Test by Participant ID."""
        # Step 1: Split people into (Train+Val) vs Test
        gssTest = GroupShuffleSplit(
            n_splits=1,
            test_size=self.testSize,
            random_state=self.randomState,
        )
        trainValIdx, testIdx = next(gssTest.split(X, y, groups))

        # Step 2: Split (Train+Val) into Train vs Val
        if self.valSize > 0:
            relativeValSize = self.valSize / (1.0 - self.testSize)
            gssVal = GroupShuffleSplit(
                n_splits=1,
                test_size=relativeValSize,
                random_state=self.randomState + 1,
            )
            subGroups = groups[trainValIdx]
            subX = X.iloc[trainValIdx] if isinstance(X, pd.DataFrame) else X[trainValIdx]
            subY = y[trainValIdx]

            trainRelIdx, valRelIdx = next(gssVal.split(subX, subY, subGroups))
            trainIdx = trainValIdx[trainRelIdx]
            valIdx = trainValIdx[valRelIdx]
        else:
            trainIdx = trainValIdx
            valIdx = np.array([], dtype=int)

        # Step 3: Hard assert check to guarantee 0% subject overlap (no data leakage!)
        trainGroups = set(groups[trainIdx])
        testGroups = set(groups[testIdx])
        valGroups = set(groups[valIdx]) if len(valIdx) > 0 else set()

        assert trainGroups.isdisjoint(testGroups), "Data leakage: Same person found in Train and Test!"
        if valGroups:
            assert trainGroups.isdisjoint(valGroups), "Data leakage: Same person found in Train and Val!"
            assert valGroups.isdisjoint(testGroups), "Data leakage: Same person found in Val and Test!"

        return {
            "train": trainIdx,
            "val": valIdx,
            "test": testIdx,
        }


def createCrossValidationSplits(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    nSplits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Creates K-fold cross validation splits where entire participants are kept in separate folds."""
    gkf = GroupKFold(n_splits=nSplits)
    splits = []
    for trainIdx, testIdx in gkf.split(X, y, groups):
        # Double check no overlap on every fold
        trainP = set(groups[trainIdx])
        testP = set(groups[testIdx])
        assert trainP.isdisjoint(testP), "Subject overlap detected in K-Fold split!"
        splits.append((trainIdx, testIdx))
    return splits


# Aliases for backward compatibility
create_cross_validation_splits = createCrossValidationSplits
