"""Feature preprocessing, scaling, and variance filtering for EEG features."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class EEGPreprocessor(BaseEstimator, TransformerMixin):
    """Pipeline component for filling missing values, removing dead features, and normalizing scale."""

    def __init__(
        self,
        scaler: str = "robust",
        scalerScope: str = "global",
        varianceThreshold: float = 1e-5,
        imputeStrategy: str = "median",
        **kwargs,
    ):
        """Initializes the preprocessor.

        Args:
            scaler: How to normalize ('robust', 'standard', 'minmax', etc.).
            scalerScope: 'global' (scale across whole training set) or 'subject' (scale per person).
            varianceThreshold: Delete features whose variance is below this cut-off.
            imputeStrategy: How to fill missing values ('median' or 'mean').
        """
        self.scaler = scaler
        self.scalerScope = kwargs.get("scaler_scope", scalerScope)
        self.scaler_scope = self.scalerScope
        self.varianceThreshold = kwargs.get("variance_threshold", varianceThreshold)
        self.variance_threshold = self.varianceThreshold
        self.imputeStrategy = kwargs.get("impute_strategy", imputeStrategy)
        self.impute_strategy = self.imputeStrategy

        self.imputer: Optional[SimpleImputer] = None
        self.selector: Optional[VarianceThreshold] = None
        self.globalScaler: Optional[BaseEstimator] = None
        self.global_scaler = None
        self.featureNames_: Optional[np.ndarray] = None
        self.feature_names_ = None

    def getScalerInstance(self) -> BaseEstimator:
        """Picks the requested normalization method from scikit-learn."""
        if self.scaler == "robust":
            # Best for noisy EEG because it uses medians instead of means
            return RobustScaler()
        elif self.scaler == "standard":
            return StandardScaler()
        elif self.scaler == "minmax":
            return MinMaxScaler()
        elif self.scaler == "quantile":
            return QuantileTransformer(output_distribution="normal", random_state=42)
        elif self.scaler == "maxabs":
            return MaxAbsScaler()
        elif self.scaler == "none":
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer(func=lambda x: x, validate=False)
        else:
            raise ValueError(f"Unknown scaler: {self.scaler}")

    _get_scaler_instance = getScalerInstance

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "EEGPreprocessor":
        """Learns the median values, dead features, and scaling centers from the training set."""
        if isinstance(X, pd.DataFrame):
            self.featureNames_ = X.columns.to_numpy()
            self.feature_names_ = self.featureNames_
            xArr = X.to_numpy()
        else:
            xArr = np.asarray(X)

        # 1. Fill missing values with the column median
        self.imputer = SimpleImputer(strategy=self.imputeStrategy)
        xImp = self.imputer.fit_transform(xArr)

        # 2. Drop constant / useless features
        self.selector = VarianceThreshold(threshold=self.varianceThreshold)
        xSel = self.selector.fit_transform(xImp)

        if self.featureNames_ is not None:
            self.featureNames_ = self.featureNames_[self.selector.get_support()]
            self.feature_names_ = self.featureNames_

        # 3. Fit scaler
        if self.scalerScope == "global":
            self.globalScaler = self.getScalerInstance()
            self.globalScaler.fit(xSel)
            self.global_scaler = self.globalScaler

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Applies the learned cleaning and scaling steps to new test data."""
        if isinstance(X, pd.DataFrame):
            xArr = X.to_numpy()
        else:
            xArr = np.asarray(X)

        # 1. Fill NaNs
        xImp = self.imputer.transform(xArr)

        # 2. Prune dead features
        xSel = self.selector.transform(xImp)

        # 3. Apply normalization
        if self.scalerScope == "global":
            xScaled = self.globalScaler.transform(xSel)
        elif self.scalerScope == "subject":
            if groups is None:
                raise ValueError("groups array required when scalerScope='subject'")
            xScaled = np.empty_like(xSel)
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                sc = self.getScalerInstance()
                xScaled[idx] = sc.fit_transform(xSel[idx])
        else:
            xScaled = xSel

        return xScaled

    def fitTransform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fits on data and returns transformed data in one step."""
        return self.fit(X, y, groups).transform(X, groups)

    fit_transform = fitTransform
