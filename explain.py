from typing import Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# SHAP for pipeline: we explain the final model using transformed features.


def _transform_inputs(pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, list]:
    pre = pipeline.named_steps["preprocess"]
    X_t = pre.transform(X)
    feat_names = pre.get_feature_names_out().tolist()
    return X_t, feat_names


def shap_global_summary(pipeline, X_background: pd.DataFrame, max_display: int = 20):
    X_t, feat_names = _transform_inputs(pipeline, X_background)
    model = pipeline.named_steps["model"]

    explainer = shap.Explainer(model, X_t, feature_names=feat_names)
    shap_values = explainer(X_t)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_t, feature_names=feat_names, show=False, max_display=max_display)
    plt.tight_layout()
    return fig


def shap_waterfall_for_single(pipeline, X_single: pd.DataFrame):
    X_t, feat_names = _transform_inputs(pipeline, X_single)
    model = pipeline.named_steps["model"]

    explainer = shap.Explainer(model, X_t, feature_names=feat_names)
    shap_values = explainer(X_t)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False, max_display=20)
    plt.tight_layout()
    return fig