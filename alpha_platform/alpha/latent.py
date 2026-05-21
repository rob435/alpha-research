"""
Module 4 - Latent Feature & Rank-Skill Classifier
==================================================

Stacking many brittle hand-tuned binary filters - ``turnover > X`` AND
``wick < Y`` AND ``illiquidity < Z`` - is fragile twice over: the gates are
mutually collinear (so the conjunction throws away far more than intended),
and a single borderline axis vetoes an otherwise textbook asset.

Instead this module collapses every continuous diagnostic into ONE smooth
probability field:

    latent_extremity_score  in  [0.0, 1.0]

If an asset has extreme rank turnover it is not blocked merely because its
close location is slightly off - the JOINT configuration is scored, not each
axis in isolation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LatentExtremityEvaluator:
    r"""Rolling latent-extremity classifier.

    Inputs
        features : Mapping[str, pd.DataFrame]
            ``{feature_name -> wide panel}``; every panel is timestamp x
            asset and the panels MUST be mutually aligned. Canonical inputs:
                rank_velocity, residual_return     (Module 1)
                illiquidity_proxy, wick_ratio      (Module 2)

    Output
        ``score()`` -> a wide [0, 1] panel of latent_extremity_score.

    Model choice
        model="isolation_forest" (default) - unsupervised. Extremity is the
            anomaly score: a joint feature configuration far from the bulk of
            the training cloud scores high. No labels required.
        model="logistic" - supervised. Requires a label panel (see
            ``build_extremity_labels``) and reads off predict_proba.

    Look-ahead protection
        The estimator is FIT only on rows whose timestamp <= ``train_end``
        and only then applied to the whole panel. Imputer and scaler
        statistics are learned exclusively on that training slice - the
        sklearn ``Pipeline`` guarantees the transforms are not re-fit at
        score time.
    """

    def __init__(self, model: str = "isolation_forest",
                 random_state: int = 0, **model_kwargs) -> None:
        if model not in ("isolation_forest", "logistic"):
            raise ValueError("model must be 'isolation_forest' or 'logistic'")
        self.model_name = model
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] | None = None
        self._panel_index: pd.Index | None = None
        self._panel_columns: pd.Index | None = None
        # (centre, scale) used to map raw anomaly scores into [0, 1];
        # calibrated on the TRAINING slice only.
        self._calibration: tuple[float, float] = (0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Panel <-> design-matrix plumbing                                   #
    # ------------------------------------------------------------------ #
    def _stack(self, features: Mapping[str, pd.DataFrame]) -> tuple[np.ndarray, pd.Index]:
        """Flatten ``{name: wide panel}`` into a 2-D design matrix.

        Row order is timestamp-major (C-order ravel): for a (T, A) panel the
        rows run (t0,a0), (t0,a1), ... (t0,aN), (t1,a0), ...  Every feature
        is raveled identically, so column k of the design matrix is feature
        ``self._feature_names[k]``. ``stack`` from pandas is deliberately
        avoided - explicit raveling is version-stable and order-deterministic.
        """
        names = sorted(features)
        reference = features[names[0]]
        self._panel_index = reference.index
        self._panel_columns = reference.columns
        self._feature_names = names

        columns = []
        for name in names:
            panel = features[name]
            if (not panel.index.equals(reference.index)
                    or not panel.columns.equals(reference.columns)):
                raise ValueError(f"feature panel '{name}' is misaligned")
            columns.append(panel.to_numpy(dtype=float).ravel())
        return np.column_stack(columns), reference.index

    def _train_mask(self, train_end) -> np.ndarray:
        """Boolean mask over design-matrix rows with timestamp <= train_end.

        The timestamp repeats once per asset, matching the timestamp-major
        ravel order used by ``_stack``.
        """
        n_assets = len(self._panel_columns)
        is_train_ts = self._panel_index <= pd.Timestamp(train_end)
        return np.repeat(np.asarray(is_train_ts), n_assets)

    # ------------------------------------------------------------------ #
    # Fit                                                                #
    # ------------------------------------------------------------------ #
    def fit(self, features: Mapping[str, pd.DataFrame], train_end,
            labels: pd.DataFrame | None = None) -> "LatentExtremityEvaluator":
        """Fit the classifier on the training slice (timestamp <= train_end)."""
        design, _ = self._stack(features)
        train_rows = design[self._train_mask(train_end)]
        # Drop rows that are entirely NaN (warm-up bars); the imputer in the
        # pipeline handles any remaining per-feature gaps.
        train_rows = train_rows[~np.isnan(train_rows).all(axis=1)]
        if train_rows.shape[0] == 0:
            raise ValueError("training slice is empty after NaN removal")

        if self.model_name == "isolation_forest":
            self._fit_isolation_forest(train_rows)
        else:
            self._fit_logistic(train_rows, features, train_end, labels)
        return self

    def _fit_isolation_forest(self, train_rows: np.ndarray) -> None:
        params = dict(n_estimators=200, contamination="auto",
                      random_state=self.random_state)
        params.update(self.model_kwargs)
        self._pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", IsolationForest(**params)),
        ])
        self._pipeline.fit(train_rows)

        # Calibrate the [0, 1] mapping on TRAINING anomaly scores only.
        # IsolationForest.score_samples: higher = more normal; negating it
        # makes "higher = more extreme". Centre/scale by the training median
        # and IQR so the logistic squashing is distribution-aware.
        raw = -self._pipeline.score_samples(train_rows)
        centre = float(np.median(raw))
        iqr = float(np.subtract(*np.percentile(raw, [75, 25])))
        scale = iqr if iqr > 1e-12 else (float(np.std(raw)) or 1.0)
        self._calibration = (centre, scale)

    def _fit_logistic(self, train_rows, features, train_end, labels) -> None:
        if labels is None:
            raise ValueError("model='logistic' requires a `labels` panel")
        label_vec = labels.to_numpy(dtype=float).ravel()[self._train_mask(train_end)]
        # Re-derive the all-NaN row mask so features and labels stay aligned.
        design = self._stack(features)[0]
        train_design = design[self._train_mask(train_end)]
        keep = ~np.isnan(train_design).all(axis=1)
        x_train = train_design[keep]
        y_train = label_vec[keep]
        usable = ~np.isnan(y_train)
        if usable.sum() == 0:
            raise ValueError("no usable (feature, label) training rows")

        params = dict(max_iter=1000, class_weight="balanced",
                      random_state=self.random_state)
        params.update(self.model_kwargs)
        self._pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
        self._pipeline.fit(x_train[usable], y_train[usable].astype(int))

    # ------------------------------------------------------------------ #
    # Score                                                              #
    # ------------------------------------------------------------------ #
    def score(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Produce the latent_extremity_score panel for every (t, asset).

        Returns a wide [0, 1] DataFrame. Warm-up rows (all features NaN)
        remain NaN rather than being imputed to a misleading mid-score.
        """
        if self._pipeline is None:
            raise RuntimeError("call fit() before score()")
        design, index = self._stack(features)
        valid = ~np.isnan(design).all(axis=1)

        result = np.full(design.shape[0], np.nan)
        if valid.any():
            x_valid = design[valid]
            if self.model_name == "isolation_forest":
                raw = -self._pipeline.score_samples(x_valid)
                centre, scale = self._calibration
                # Logistic squash -> guaranteed (0, 1), monotone in extremity.
                result[valid] = 1.0 / (1.0 + np.exp(-(raw - centre) / scale))
            else:
                result[valid] = self._pipeline.predict_proba(x_valid)[:, 1]

        wide = result.reshape(len(self._panel_index), len(self._panel_columns))
        return pd.DataFrame(wide, index=self._panel_index,
                            columns=self._panel_columns)

    # ------------------------------------------------------------------ #
    # Optional label construction for the supervised path                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_extremity_labels(forward_returns: pd.DataFrame,
                               quantile: float = 0.9) -> pd.DataFrame:
        """Binary 'extreme outcome' labels for the logistic model.

        label = 1 if the asset's forward return sits in the top OR bottom
        tail of its cross-section at that bar, else 0.

        EMBARGO WARNING (purge gap)
            A label at time t depends on return(t+N). If logistic is trained
            on rows up to ``train_end``, those labels peek up to N bars past
            ``train_end``. The evaluation/test set MUST therefore begin at
            ``train_end + N`` bars so feature-period leakage cannot occur.
        """
        hi = forward_returns.quantile(quantile, axis=1)
        lo = forward_returns.quantile(1.0 - quantile, axis=1)
        extreme = forward_returns.ge(hi, axis=0) | forward_returns.le(lo, axis=0)
        return extreme.astype(float).where(forward_returns.notna())
