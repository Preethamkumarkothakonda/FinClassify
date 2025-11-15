# app.py
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

# minimum number of times a custom label must appear in feedback
# before it is promoted to a real taxonomy category
PROMOTION_THRESHOLD = 5


# =========================
# TAXONOMY HELPERS
# =========================
def load_taxonomy():
    """Load category taxonomy from JSON config."""
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    return data["categories"]


def save_taxonomy(categories):
    """Save category list back to JSON."""
    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)
    data["categories"] = categories
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)


CATEGORY_LIST = load_taxonomy()


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()

    # normalize common finance acronyms / tokens
    text = text.replace("pos", " pointofsale ")
    text = text.replace("txn", " transaction ")
    text = text.replace("neft", " banktransfer ")
    text = text.replace("upi", " upipayment ")

    # remove long numeric IDs (references, auth codes)
    text = re.sub(r"\b\d{4,}\b", " ", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# CACHED LOADERS
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    vect = joblib.load(VECT_FILE)
    return model, vect


@st.cache_resource(show_spinner=False)
def make_explainer(_model: LogisticRegression, _vect: TfidfVectorizer):
    """Construct a SHAP explainer for text inputs."""
    masker = shap.maskers.Text(tokenizer=str.split)
    explainer = shap.Explainer(_model, masker)
    feature_names = np.array(_vect.get_feature_names_out())
    return explainer, feature_names


# =========================
# FEEDBACK SAVING
# =========================
def append_feedback(text: str, predicted_core: str, corrected_core: str, corrected_ui: str):
    row = {
        "description": text,
        "predicted": predicted_core,
        "corrected": corrected_core,   # internal label used for training
        "corrected_ui": corrected_ui,  # what user actually selected/typed
        "timestamp": str(pd.Timestamp.now()),
    }

    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(FEEDBACK_FILE, index=False)


# =========================
# RETRAIN (HYBRID CATEGORY PROMOTION)
# =========================
def retrain_model():
    """
    Hybrid behavior:
    - base taxonomy is read from taxonomy.json
    - feedback 'corrected_ui' values are counted
    - any custom label with count >= PROMOTION_THRESHOLD becomes a real category
      and is added to taxonomy.json
    - training uses: base categories + promoted categories
    - all other labels are coerced to 'Others'
    """
    base_categories = load_taxonomy()
    current_categories = list(base_categories)

    df_main = pd.read_csv(DATA_FILE)
    if "description" not in df_main.columns or "category" not in df_main.columns:
        raise ValueError("DATA_FILE must contain 'description' and 'category' columns.")

    # Clean main data
    df_main["clean"] = df_main["description"].apply(clean_text)

    # Coerce any label not in base taxonomy to 'Others'
    df_main["category"] = df_main["category"].apply(
        lambda c: c if c in base_categories else "Others"
    )

    df_all = df_main[["clean", "category"]].copy()

    # Process feedback if available
    if os.path.exists(FEEDBACK_FILE):
        df_fb = pd.read_csv(FEEDBACK_FILE)
        if not df_fb.empty:
            # Count how many times each corrected_ui appears
            counts = df_fb["corrected_ui"].value_counts()

            promoted = []
            for label, count in counts.items():
                label = str(label)
                if (
                    label not in base_categories
                    and label.lower() != "others"
                    and count >= PROMOTION_THRESHOLD
                ):
                    promoted.append(label)

            # If any new categories are promoted, update taxonomy
            if promoted:
                for nc in promoted:
                    if nc not in current_categories:
                        current_categories.append(nc)
                save_taxonomy(current_categories)

            # Re-load current categories to ensure consistency
            current_categories = load_taxonomy()

            # Clean feedback descriptions
            df_fb["clean"] = df_fb["description"].apply(clean_text)

            # For training: if corrected_ui is in current categories, use it,
            # otherwise fall back to 'Others'
            df_fb["category"] = df_fb["corrected_ui"].apply(
                lambda c: c if c in current_categories else "Others"
            )

            df_all = pd.concat(
                [df_all, df_fb[["clean", "category"]]], ignore_index=True
            )

    # Vectorize + train model
    vect = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        analyzer="word",
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )
    X = vect.fit_transform(df_all["clean"])
    y = df_all["category"]

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        n_jobs=-1,
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
# EXPLANATION (SHAP + FALLBACK)
# =========================
def explain_features(explainer, feature_names, X_vector, predicted_label, model, top_k=10):
    """
    Try SHAP explanation first; if it fails or explainer is None,
    fall back to simple coef * tfidf attribution.
    Returns [(feature, contribution, tfidf_value), ...]
    """
    X_dense = X_vector.toarray().reshape(-1)
    nonzero_idx = np.where(X_dense != 0)[0]

    # Try SHAP
    if explainer is not None:
        try:
            sh_out = explainer(X_vector)
            vals = getattr(sh_out, "values", None)

            if isinstance(vals, dict):
                # SHAP returned a dict-like mapping
                if predicted_label in vals:
                    arr = np.array(vals[predicted_label])
                else:
                    arr = np.array(list(vals.values())[0])
            else:
                arr = np.array(vals)

            # Flatten shapes
            if arr.ndim == 3:
                # (samples, features, classes)
                class_names = list(getattr(sh_out, "output_names", []))
                if class_names and predicted_label in class_names:
                    class_index = class_names.index(predicted_label)
                else:
                    class_index = 0
                sv = arr[0, :, class_index]
            elif arr.ndim == 2:
                # (1, features) or (classes, features)
                if arr.shape[0] == 1:
                    sv = arr[0, :]
                else:
                    class_names = list(getattr(sh_out, "output_names", []))
                    if class_names and arr.shape[0] == len(class_names):
                        class_index = class_names.index(predicted_label)
                        sv = arr[class_index, :]
                    else:
                        sv = arr.ravel()
            else:
                sv = arr.ravel()

            sv = np.array(sv).reshape(-1)

            contribs = [
                (feature_names[i], float(sv[i]), float(X_dense[i]))
                for i in nonzero_idx
            ]
            contribs.sort(key=lambda t: abs(t[1]), reverse=True)
            return contribs[:top_k]

        except Exception:
            pass  # fall through to coef-based fallback

    # Fallback: coefficient * tfidf
    if model is None:
        return []

    if hasattr(model, "classes_"):
        try:
            class_index = list(model.classes_).index(predicted_label)
        except ValueError:
            class_index = 0
    else:
        class_index = 0

    coef = np.array(model.coef_)
    if coef.ndim == 1:
        coef_vec = coef
    else:
        coef_vec = coef[class_index]

    contrib_array = coef_vec * X_dense
    contribs = [
        (feature_names[i], float(contrib_array[i]), float(X_dense[i]))
        for i in nonzero_idx
    ]
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return contribs[:top_k]


# =========================
# UI
# =========================
st.set_page_config(page_title="FinClassify", page_icon="🧩", layout="centered")
st.title("FinClassify")
st.caption("Config-driven taxonomy, hybrid custom categories, feedback loop + explainability.")

defaults = {"last_text": "Starbucks Coffee POS TXN", "pred": None, "conf": None, "X_vec": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Load artifacts
model, vect = load_artifacts()

# Reload taxonomy each run (in case retrain promoted new categories)
CATEGORY_LIST = load_taxonomy()

# ====== Transaction Input ======
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
        # Clear caches and reload artifacts
        try:
            load_artifacts.clear()
            make_explainer.clear()
        except Exception:
            pass
        st.success("Retrained model and updated taxonomy if needed.")
        st.rerun()


# ====== Feedback UI ======
if st.session_state.pred is not None:
    st.markdown(
        f"Last prediction: *{st.session_state.pred}* — {st.session_state.conf*100:.2f}%"
    )

    with st.form("feedback_form", clear_on_submit=False):
        st.write("### Submit Feedback")

        default_idx = CATEGORY_LIST.index(st.session_state.pred) if st.session_state.pred in CATEGORY_LIST else CATEGORY_LIST.index("Others")

        corrected_select = st.selectbox(
            "Choose a correct label from existing taxonomy:",
            CATEGORY_LIST,
            index=default_idx,
        )

        custom_text = st.text_input(
            "Or type a NEW custom category (hybrid: promoted after enough feedback):",
            value="",
        )

        submitted = st.form_submit_button("Save Feedback")

        if submitted:
            custom_text = custom_text.strip()

            if custom_text:
                # Hybrid behavior: new category is stored as UI label but internally starts as 'Others'
                corrected_ui = custom_text
                corrected_core = "Others"
            else:
                corrected_ui = corrected_select
                corrected_core = corrected_select

            append_feedback(
                st.session_state.last_text,
                st.session_state.pred,
                corrected_core,
                corrected_ui,
            )
            st.success(
                f"Feedback saved. Internal label: {corrected_core} | UI label: {corrected_ui}."
            )


# ====== Explanation Section ======
st.subheader("Why this prediction?")
if st.session_state.X_vec is not None and st.session_state.pred is not None:
    if st.button("Compute Explanation"):
        try:
            explainer, feature_names = make_explainer(model, vect)
        except Exception:
            explainer = None
            feature_names = np.array(vect.get_feature_names_out())

        X_vec = st.session_state.X_vec
        top_feats = explain_features(
            explainer,
            feature_names,
            X_vec,
            predicted_label=st.session_state.pred,
            model=model,
            top_k=10,
        )
        if top_feats:
            df_shap = pd.DataFrame(
                [
                    {
                        "feature": f,
                        "contribution": float(sv),
                        "tfidf_value": float(tv),
                    }
                    for f, sv, tv in top_feats
                ]
            )
            st.dataframe(df_shap)
        else:
            st.info("No feature attributions available for this input.")
    else:
        st.caption("Click the button to view feature-level attributions (may be slower).")
else:
    st.info("Predict a transaction first to see explanations.")


# ====== Feedback Table ======
st.subheader("Feedback Records")
if os.path.exists(FEEDBACK_FILE):
    df_fb = pd.read_csv(FEEDBACK_FILE)
    st.dataframe(df_fb.tail(20))
else:
    st.info("No feedback submitted yet.")