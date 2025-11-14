# app_phase2_fixed_with_others.py
import os
import re
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CONFIG / PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "transaction_classifier_model.pkl")
VECT_FILE = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
DATA_FILE = os.path.join(BASE_DIR, "synthetic_transactions.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "taxonomy.json")


# =========================
# TAXONOMY LOADER
# =========================
def load_taxonomy():
    """Load category taxonomy from JSON config."""
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    return data["categories"]


CATEGORY_LIST = load_taxonomy()
UI_LABELS = CATEGORY_LIST  # UI shows the same fixed set


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# MAPPING HELPERS
# =========================
def to_core_label(ui_label: str) -> str:
    """Map any UI label not in taxonomy to 'Others'."""
    return ui_label if ui_label in CATEGORY_LIST else "Others"


# =========================
# DATA HELPERS
# =========================
def filter_to_fixed_categories(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return df[df[label_col].isin(CATEGORY_LIST)].copy()


# =========================
# CACHED LOADERS
# =========================
@st.cache_resource
def load_artifacts():
    model: LogisticRegression = joblib.load(MODEL_FILE)
    vect: TfidfVectorizer = joblib.load(VECT_FILE)
    return model, vect


@st.cache_resource
def make_explainer(_model: LogisticRegression, _vect: TfidfVectorizer):
    # Create masker for sparse input
    masker = shap.maskers.Text(tokenizer=str.split)

    # Explainer for linear model
    explainer = shap.Explainer(_model, masker)
    feature_names = np.array(_vect.get_feature_names_out())
    return explainer, feature_names


# =========================
# FEEDBACK
# =========================
def append_feedback(text: str, predicted_core: str, corrected_core: str, corrected_ui: str):
    row = {
        "description": text,
        "predicted": predicted_core,
        "corrected": corrected_core,   # training label (in fixed set)
        "corrected_ui": corrected_ui,  # optional UI text for audit
        "timestamp": str(pd.Timestamp.now()),
    }

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(FEEDBACK_FILE, index=False)


# =========================
# TRAIN / RETRAIN (fixed labels + Others)
# =========================
def retrain_model():
    df_main = pd.read_csv(DATA_FILE)
    if "description" not in df_main.columns or "category" not in df_main.columns:
        raise ValueError("DATA_FILE must have 'description' and 'category' columns.")

    df_main["clean"] = df_main["description"].apply(clean_text)

    # Coerce any non-fixed label in base data to Others (guard)
    df_main["category"] = df_main["category"].apply(
        lambda c: c if c in CATEGORY_LIST else "Others"
    )
    df_all = df_main[["clean", "category"]].copy()

    # Add feedback data if present
    if os.path.exists(FEEDBACK_FILE):
        df_fb = pd.read_csv(FEEDBACK_FILE)
        if not df_fb.empty:
            df_fb = df_fb.rename(columns={"corrected": "category"})
            df_fb["clean"] = df_fb["description"].apply(clean_text)
            df_all = pd.concat(
                [df_all, df_fb[["clean", "category"]]], ignore_index=True
            )

    vect = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    X = vect.fit_transform(df_all["clean"])
    y = df_all["category"]

    model = LogisticRegression(
        max_iter=2000, multi_class="auto", class_weight="balanced"
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vect, VECT_FILE)


# =========================
# PREDICT
# =========================
def predict(model: LogisticRegression, vect: TfidfVectorizer, text: str):
    cleaned = clean_text(text)
    X = vect.transform([cleaned])
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    pred = model.classes_[idx]
    conf = float(proba[idx])
    return pred, conf, X


# =========================
# SHAP UTIL
# =========================
def shap_explain_single(explainer, feature_names, X_vector, class_index=None, top_k=10):
    shap_values = explainer.shap_values(X_vector)

    if isinstance(shap_values, list):
        if class_index is None:
            sums = [np.sum(np.abs(sv)) for sv in shap_values]
            class_index = int(np.argmax(sums))
        sv = np.array(shap_values[class_index]).reshape(-1)
    else:
        sv = np.array(shap_values).reshape(-1)

    X_dense = X_vector.toarray().reshape(-1)
    nonzero_idx = np.where(X_dense != 0)[0]
    contribs = [(feature_names[i], sv[i], X_dense[i]) for i in nonzero_idx]
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return contribs[:top_k]


# =========================
# UI
# =========================
st.set_page_config(
    page_title="FinClassify — Phase 2 (+Others)",
    page_icon="🧩",
    layout="centered",
)
st.title("FinClassify — Phase 2 (+ Others)")
st.caption("Config-driven taxonomy, Others bucket, feedback loop + SHAP explainability.")

defaults = {
    "last_text": "Starbucks Coffee POS TXN",
    "pred": None,
    "conf": None,
    "X_vec": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

model, vect = load_artifacts()

st.subheader("Enter Transaction")
text = st.text_input("Transaction description", value=st.session_state.last_text)

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    if st.button("Predict"):
        st.session_state.last_text = text
        pred, conf, X_vec = predict(model, vect, text)
        st.session_state.pred = pred
        st.session_state.conf = conf
        st.session_state.X_vec = X_vec
        st.success(f"Prediction: {pred} — Confidence: {conf*100:.2f}%")

with c2:
    if st.button("Clear"):
        for k in ["pred", "conf", "X_vec"]:
            st.session_state[k] = None
        st.info("Cleared current prediction.")

with c3:
    if st.button("Retrain with Feedback"):
        retrain_model()
        load_artifacts.clear()
        make_explainer.clear()
        model, vect = load_artifacts()
        st.success("Retrained and reloaded artifacts.")
        st.rerun()


# =========================
# FEEDBACK UI
# =========================
if st.session_state.pred is not None:
    st.markdown(
        f"*Last prediction:* {st.session_state.pred} — {st.session_state.conf*100:.2f}%"
    )

    with st.form("feedback_form", clear_on_submit=False):
        st.write("### Submit Feedback")

        # Safe default index even if taxonomy changed
        if st.session_state.pred in CATEGORY_LIST:
            default_idx = CATEGORY_LIST.index(st.session_state.pred)
        else:
            default_idx = CATEGORY_LIST.index("Others")

        corrected_select = st.selectbox(
            "Choose a label:", CATEGORY_LIST, index=default_idx
        )
        custom_text = st.text_input(
            "Or type a custom label (will be saved as Others):", value=""
        )

        corrected_ui = custom_text.strip() if custom_text.strip() else corrected_select
        corrected_core = to_core_label(corrected_ui)

        submitted = st.form_submit_button("Save Feedback")
        if submitted:
            append_feedback(
                st.session_state.last_text,
                st.session_state.pred,
                corrected_core,
                corrected_ui,
            )
            st.success(
                f"Feedback saved (stored as {corrected_core}; UI entered: {corrected_ui})."
            )


# =========================
# SHAP EXPLANATIONS
# =========================
st.subheader("Why this prediction?")
try:
    explainer, feature_names = make_explainer(model, vect)
    X_vec = st.session_state.X_vec
    if X_vec is not None and st.session_state.pred is not None:
        class_index = list(model.classes_).index(st.session_state.pred)
        top_feats = shap_explain_single(
            explainer, feature_names, X_vec, class_index=class_index, top_k=10
        )
        if top_feats:
            df_shap = pd.DataFrame(
                [
                    {
                        "feature": f,
                        "shap_value": float(sv),
                        "tfidf_value": float(tv),
                    }
                    for f, sv, tv in top_feats
                ]
            )
            st.dataframe(df_shap)
        else:
            st.info("No non-zero TF-IDF features found for this input.")
    else:
        st.info("Predict once to see SHAP attributions.")
except Exception as e:
    st.warning(f"SHAP explanation not available: {e}")


# =========================
# FEEDBACK TABLE
# =========================
st.subheader("Feedback Records")
if os.path.exists(FEEDBACK_FILE):
    df_fb = pd.read_csv(FEEDBACK_FILE)
    st.dataframe(df_fb.tail(20))
else:
    st.info("No feedback submitted yet.")