from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import joblib

def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols


def build_pipeline(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    cat_cols, num_cols = infer_feature_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipe, cat_cols, num_cols


def train_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict:
    # 1. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pipeline, cat_cols, num_cols = build_pipeline(X_train)

    # Calculate class weight for handling imbalance
    class_weights = y_train.value_counts(normalize=True)
    scale_pos_weight = class_weights[0] / class_weights[1]

    # 2. Add Class Imbalance Handling
    pipeline.set_params(model__scale_pos_weight=scale_pos_weight)

    # 3. Use RandomizedSearchCV for efficiency and set scoring to ROC AUC
    param_grid = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__gamma": [0, 0.1, 0.2],
        "model__min_child_weight": [1, 2, 3]
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=50,  # 4. Increased iterations for broader search
        cv=5,      # 5. Increased cross-validation folds for more stable results
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X_train, y_train)

    best_pipeline = random_search.best_estimator_

    # 6. Tune Prediction Threshold for better Precision/Recall
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Choose a custom threshold based on business needs. For example, to prioritize recall:
    # y_pred = (y_proba >= 0.3).astype(int) 
    
    # The default threshold of 0.5 is used here
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    
    # Get a list of the best hyperparameters
    best_params = random_search.best_params_

    # Permutation importance on a small sample to keep it snappy
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    perm = permutation_importance(best_pipeline, X_test.iloc[sample_idx], y_test.iloc[sample_idx], n_repeats=5, random_state=42)

    # Get feature names from the correct pipeline step
    # Ensure the step name 'preprocess' matches the name given in build_pipeline
    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    importances = sorted(zip(feature_names, perm.importances_mean), key=lambda x: x[1], reverse=True)
    
    return {
        "pipeline": best_pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": {
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "best_params": best_params,
        },
        "feature_importance": importances,
    }


def save_model(pipeline: Pipeline, path: str = "churn_model.pkl") -> None:
    joblib.dump(pipeline, path)


def load_model(path: str = "churn_model.pkl") -> Pipeline:
    return joblib.load(path)


def predict_proba(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)[:, 1]