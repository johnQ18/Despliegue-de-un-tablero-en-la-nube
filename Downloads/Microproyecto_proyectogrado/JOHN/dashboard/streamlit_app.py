import os
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Churn Predictor | Olist", layout="wide")

DATASET_PATH = "outputs/tables/customer_churn_dataset_v2.csv"
MODEL_PATH   = "outputs/models/best_model_logreg.joblib"
THRESH_PATH  = "outputs/tables/best_threshold_logreg.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

@st.cache_data
def load_threshold():
    df = pd.read_csv(THRESH_PATH)
    return float(df.loc[0, "best_threshold"])

st.title("Churn Predictor | Olist E-commerce")

# Validación simple
for p in [DATASET_PATH, MODEL_PATH, THRESH_PATH]:
    if not os.path.exists(p):
        st.error(f"No encontré: {p}  (Ejecuta el notebook para generar outputs/)")
        st.stop()

df = load_data()
model = load_model()
thr = load_threshold()

# Sidebar
st.sidebar.header("Filtros & selección")
state_options = ["(Todos)"] + sorted(df["customer_state"].dropna().unique().tolist())
state_filter = st.sidebar.selectbox("Estado", state_options)

work = df.copy()
if state_filter != "(Todos)":
    work = work[work["customer_state"] == state_filter]

customer_ids = work["customer_unique_id"].astype(str).tolist()
selected_id = st.sidebar.selectbox("Cliente (customer_unique_id)", customer_ids)

# Macro
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Distribución churn (label)")
    counts = work["churned"].value_counts().sort_index()
    st.write(counts.rename({0: "No churn (recompra)", 1: "Churn"}))
    st.caption("Nota: dataset desbalanceado; por eso usamos PR-AUC/F1 de no-churn.")

with col2:
    st.subheader("Tasa de churn por estado (Top 10)")
    churn_by_state = (work.groupby("customer_state")["churned"]
                      .mean()
                      .sort_values(ascending=False)
                      .head(10))
    st.bar_chart(churn_by_state)

st.divider()

# Panel individual
row = df[df["customer_unique_id"].astype(str) == str(selected_id)].copy()
X = row.drop(columns=["customer_unique_id", "last_purchase_pre", "churned"])
p_churn = float(model.predict_proba(X)[0, 1])
pred = int(p_churn >= thr)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.subheader("Predicción")
    st.metric("Probabilidad de churn", f"{p_churn*100:.2f}%")
    st.caption(f"Umbral operativo: {thr:.2f}")
    st.metric("Riesgo", "ALTO" if pred == 1 else "BAJO")

with c2:
    st.subheader("Features del cliente")
    show_cols = ["Recency","Frequency","Monetary","avg_review_score","avg_delivery_days","avg_late_days",
                 "avg_num_items","avg_price_sum","avg_freight_sum","customer_state"]
    st.dataframe(row[show_cols].T)

with c3:
    st.subheader("Export")
    st.json({
        "customer_unique_id": str(selected_id),
        "p_churn": p_churn,
        "threshold": thr,
        "pred_label": pred
    })