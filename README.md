
---
# **README.md**

# **FinClassify – Autonomous Transaction Categorization System**

### *GHCI Hackathon 2025 Submission*

---

## **1. Overview**

**FinClassify** is an end-to-end, fully autonomous transaction categorization system built for the GHCI Hackathon.
It converts raw financial transaction descriptions into predefined spending categories with:

* High accuracy (Macro F1 = **1.00**)
* Fully offline inference
* Config-driven taxonomy
* Human-in-loop correction
* SHAP explainability
* Robust text preprocessing

The system is designed to be **transparent, configurable, responsible**, and meets all required hackathon criteria.

---

## **2. Features**

### ✔ End-to-End Offline Categorization

Processes text → cleans → vectorizes → classifies → outputs label + confidence.

### ✔ Configurable Taxonomy (`taxonomy.json`)

Admin can modify categories **without touching code**.

### ✔ Feedback Loop

Users correct predictions → saved to `feedback.csv` → retraining enabled.

### ✔ Explainability (SHAP)

Shows top contributing features for each prediction.

### ✔ Noise-Resistant Text Cleaning

Handles uppercase, punctuation, messy merchant names, numeric junk, etc.

### ✔ High Accuracy

Synthetic balanced dataset achieves **1.00 macro F1**.

---

## **3. Project Architecture**

```
Raw Text
   │
   ▼
Text Cleaner → TF-IDF Vectorizer → Logistic Regression Model
   │
   ├──► Prediction + Confidence
   ├──► SHAP Explainability
   ▼
Feedback & Corrections → feedback.csv → Retraining
```

---

## **4. Installation**

### **Clone Repository**

```
git clone <repo_link>
cd project_folder
```

### **Install Dependencies**

```
pip install -r requirements.txt
```

---

## **5. Run the App**

```
streamlit run app.py
```

---

## **6. Configurable Taxonomy**

Edit categories in:

```
taxonomy.json
```

Example:

```json
{
  "categories": [
    "Food & Dining",
    "Groceries",
    "Fuel",
    "Shopping",
    "Entertainment",
    "Bills & Utilities",
    "Healthcare",
    "Travel",
    "Education",
    "Rent",
    "Insurance",
    "Transfers",
    "Others"
  ]
}
```

App auto-updates labels and model mapping.

---

## **7. Feedback Loop Explained**

1. User enters transaction text
2. Model predicts category
3. User corrects if needed
4. Correction saved in `feedback.csv`
5. Retraining uses all feedback to improve model

Example saved feedback row:

```
UBER TRIP 23, Travel → corrected to Taxi
```

Retrained model saved as new `.pkl`.

---

## **8. Explainability (SHAP)**

SHAP shows:

* Keywords influencing prediction
* Direction of influence
* Strength of contribution

Example features:

| token     | shap_value | tfidf |
| --------- | ---------- | ----- |
| starbucks | 0.41       | 0.92  |
| coffee    | 0.31       | 0.81  |

---

### **Macro F1:**

```
1.00
```

### **Accuracy:**

```
1.00
```

Per-class F1 ranges 0.99–1.00.

---

Includes:

* No personal data
* No external APIs
* Balanced dataset
* SHAP transparency
* Human oversight
* Ethical model retraining

---

## **12. File Structure**

```
/root
│── app.py
│── taxonomy.json
│── synthetic_transactions.csv
│── feedback.csv
│── transaction_classifier_model.pkl
│── tfidf_vectorizer.pkl
│── README.md
│── DATASET.md
│── EVALUATION.md
│── RESPONSIBLE_AI.md
└── images/
```

---

## **13. Future Enhancements**

* BERT-based transformer model
* Auto-balancing category engine
* Live banking feed integration
* Admin analytics dashboard

---

## **14. Conclusion**

FinClassify meets **all GHCI Hackathon** requirements:

| Requirement               | Status   |
| ------------------------- | -------- |
| End-to-end categorization | ✔        |
| No external APIs          | ✔        |
| Macro F1 ≥ 0.90           | ✔ (1.00) |
| Explainability            | ✔        |
| Feedback loop             | ✔        |
| Responsible AI            | ✔        |
| Dataset documentation     | ✔        |
| Editable taxonomy         | ✔        |

