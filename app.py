import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_csv, clean_telco, split_X_y
from model import train_evaluate, predict_proba, save_model, load_model
from explain import shap_global_summary, shap_waterfall_for_single
from utils import compute_kpis, plot_churn_pie, plot_by_category
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn App", layout="wide")

st.title("📉 Customer Churn – Interactive App")
st.caption("Telco dataset demo – upload CSV or use a cached training run. Includes KPIs, training, predictions, SHAP explanations, and segmentation.")

# ================= Load Model and Default Data =================
MODEL_PATH = "churn_model.pkl"
DEFAULT_DATA_PATH = "telco_churn.csv"

@st.cache_resource
def load_model_cached():
    return joblib.load(MODEL_PATH)

model = "churn_model.pkl"

# Load default dataset
default_data = pd.read_csv(DEFAULT_DATA_PATH)


# --- Sidebar ---
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])  # e.g., WA_Fn-UseC_-Telco-Customer-Churn.csv

@st.cache_data(show_spinner=False)
def _load_and_clean(file):
    df_raw = load_csv(file)
    df = clean_telco(df_raw)
    return df

if uploaded:
    df = _load_and_clean(uploaded)
    st.sidebar.success("✅ Custom dataset loaded")
else:
    st.info("ℹ️ Using default Telco dataset")
    df = clean_telco(default_data)

# Tabs
overview, train_tab, predict_tab, explain_tab, segment_tab = st.tabs([
    "Overview", "Train & Evaluate", "Customer Prediction", "Explainability", "Segmentation"
])

# ===== Overview =====
with overview:
    st.subheader("📊 Dataset Overview & Business KPIs")
    if df is None:
        st.stop()

    st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

    kpis = compute_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{kpis['customers']:,}")
    c2.metric("Churn Rate", f"{(kpis['churn_rate'] or 0):.1f}%")
    c3.metric("Monthly Revenue (sum)", f"${kpis['monthly_revenue']:.0f}")
    c4.metric("Annual Revenue at Risk", f"${kpis['annual_rev_at_risk']:.0f}")

    # --- Churn Distribution Pie Chart ---
    if "Churn" in df.columns:
        st.subheader("📊 Churn Distribution")
        churn_counts = df["Churn"].value_counts()
        fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index,
                        title="Churn Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Churn by Category ---
    st.subheader("📊 Churn by Category")

    # Select categorical columns with limited unique values
    cat_cols = [
        c for c in df.columns 
        if df[c].dtype == "object" 
        and c != "Churn" 
        and df[c].nunique() <= 20   # drop columns with too many unique values
    ]

    if cat_cols:
        col_select = st.selectbox("Select categorical column", cat_cols)
        fig_bar = px.histogram(
            df, 
            x=col_select, 
            color="Churn" if "Churn" in df.columns else None,
            barmode="group", 
            title=f"Churn by {col_select}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No categorical columns with limited unique values to display for churn breakdown")


# ===== Train & Evaluate =====
with train_tab:
    st.subheader("🧠 Train Model & Evaluate")
    if df is None:
        st.stop()

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    X, y = split_X_y(df)
    result = train_evaluate(X, y, test_size=float(test_size))
    pipeline = result["pipeline"]

    # Metrics
    m = result["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")

    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(m["confusion_matrix"], index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    # Feature importance (permutation)
    st.write("### Top Features (Permutation Importance)")
    topk = st.slider("Show top K", 5, 30, 15)
    fi = result["feature_importance"][:topk]
    st.dataframe(pd.DataFrame(fi, columns=["Feature", "Importance"]))

    # Save model
    if st.button("💾 Save trained model"):
        save_model(pipeline)
        st.success("Model saved to churn_model.pkl")

# ===== Customer Prediction =====
with predict_tab:
    st.subheader("🔮 Customer-Level Prediction")

    # Try to load saved model, else fallback to train quickly from df
    model = None
    try:
        model = load_model("churn_model.pkl")
        st.caption("Loaded saved model: churn_model.pkl")
    except Exception:
        if df is not None:
            X, y = split_X_y(df)
            model = train_evaluate(X, y)["pipeline"]
            st.caption("No saved model found – trained a quick model from uploaded data.")
        else:
            st.warning("Upload data or train a model first.")
            st.stop()

    # Build a dynamic form from df columns
    if df is None:
        st.stop()

    X_all, _ = split_X_y(df)
    obj_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
    num_cols = [c for c in X_all.columns if X_all[c].dtype != "object"]

    with st.form("cust_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}
        for i, col in enumerate(num_cols):
            with (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3):
                val = float(st.number_input(col, value=float(X_all[col].median())))
                inputs[col] = val
        for i, col in enumerate(obj_cols):
            with (c1 if i % 3 == 0 else c2 if i % 3 == 1 else c3):
                options = sorted([str(x) for x in X_all[col].dropna().unique().tolist()])
                default = options[0] if options else ""
                val = st.selectbox(col, options=options, index=0 if default in options else 0)
                inputs[col] = val
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        x_single = pd.DataFrame([inputs])
        proba = float(predict_proba(model, x_single))
        st.metric("Churn Probability", f"{proba:.2%}")

        if proba >= 0.7:
            st.warning("⚠ High risk – recommend yearly contract discount or loyalty offer.")
        elif proba >= 0.4:
            st.info("🟡 Medium risk – flag for retention team follow-up.")
        else:
            st.success("🟢 Low risk – no intervention needed.")

        st.session_state["last_input"] = x_single
        st.session_state["last_proba"] = proba

# ===== Explainability =====
with explain_tab:
    st.subheader("🧠 Explain Predictions (SHAP)")

    # Need a trained model and some data
    model = None
    try:
        model = load_model("churn_model.pkl")
    except Exception:
        if df is not None:
            X, y = split_X_y(df)
            model = train_evaluate(X, y)["pipeline"]
    if model is None or df is None:
        st.warning("Train/load a model and upload data to view explanations.")
        st.stop()

    X, y = split_X_y(df)

    st.write("Global summary (top drivers of churn) – computed on a background sample for speed.")
    bg_size = st.slider("Background sample size", 50, 1000, 200, 50)
    bg = X.sample(n=min(bg_size, len(X)), random_state=42)
    fig = shap_global_summary(model, bg)
    st.pyplot(fig, clear_figure=True)

    st.write("\nSingle-customer explanation (waterfall):")
    if "last_input" in st.session_state:
        one = st.session_state["last_input"]
    else:
        one = X.sample(1, random_state=42)
    fig2 = shap_waterfall_for_single(model, one)
    st.pyplot(fig2, clear_figure=True)

# ===== Segmentation =====
with segment_tab:
    st.subheader("🧩 Customer Segmentation (K-Means + PCA)")
    if df is None:
        st.stop()

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer

    X, y = split_X_y(df)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], verbose_feature_names_out=False)

    n_clusters = st.slider("Clusters", 2, 8, 4)

    pipe = SkPipeline([
        ("pre", pre),
        ("pca", PCA(n_components=2, random_state=42)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
    ])

    Z = pipe.fit_transform(X)
    labels = pipe.named_steps["kmeans"].labels_

    plot_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Cluster": labels, "Churn": y.values})

    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", symbol="Churn", title="Customer Segmentation")
    st.plotly_chart(fig, use_container_width=True)
