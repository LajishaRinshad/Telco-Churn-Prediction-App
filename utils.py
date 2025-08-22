import numpy as np
import pandas as pd
import plotly.express as px


def compute_kpis(df: pd.DataFrame):
    n_customers = len(df)
    churn_rate = None
    if "Churn" in df.columns:
        churn_rate = (df["Churn"].mean()) * 100  # assumes 1/0
    monthly_rev = df.get("MonthlyCharges", pd.Series([0]*n_customers)).sum()
    annual_rev_at_risk = 0.0
    if "Churn" in df.columns and "MonthlyCharges" in df.columns:
        annual_rev_at_risk = (df.loc[df["Churn"] == 1, "MonthlyCharges"].sum()) * 12

    return {
        "customers": n_customers,
        "churn_rate": churn_rate,
        "monthly_revenue": monthly_rev,
        "annual_rev_at_risk": annual_rev_at_risk,
    }


def plot_churn_pie(df: pd.DataFrame):
    if "Churn" not in df.columns:
        return None
    counts = df["Churn"].value_counts().rename({0: "No", 1: "Yes"}).reset_index()
    counts.columns = ["Churn", "Count"]
    fig = px.pie(counts, names="Churn", values="Count", hole=0.3, title="Churn Distribution")
    return fig


def plot_by_category(df: pd.DataFrame, col: str):
    if col not in df.columns or "Churn" not in df.columns:
        return None
    tmp = df.groupby(col)["Churn"].mean().reset_index()
    tmp["Churn"] = tmp["Churn"] * 100
    fig = px.bar(tmp, x=col, y="Churn", title=f"Churn Rate by {col} (%)", text="Churn")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="Churn %", xaxis_title=col)
    return fig