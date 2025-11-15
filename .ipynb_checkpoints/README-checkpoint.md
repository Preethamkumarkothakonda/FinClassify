FinClassify – Autonomous Transaction Categorization System
GHCI Hackathon 2024 Submission
1. Overview

FinClassify is an end-to-end, fully autonomous transaction categorization system built for the GHCI Hackathon.
It transforms raw financial transaction descriptions into predefined spending categories with:

High accuracy (Macro F1 = 1.00)

Fully offline inference (no external APIs)

Config-driven taxonomy (editable by admin)

Human-in-the-loop correction and retraining

SHAP explainability

Robust preprocessing & noise handling

The system is designed to be transparent, configurable, and responsible, fulfilling all hackathon requirements.

2. Features
✔ End-to-End Offline Categorization

Takes raw text → cleans → vectorizes → classifies → returns label + confidence.

✔ Configurable Taxonomy via taxonomy.json

Admin can change categories without editing code.

✔ Feedback Loop

Users can correct predictions → corrections saved in feedback.csv → model retrains.

✔ Explainability (SHAP)

Users see why the model predicted a category.

✔ Fully Local Pipeline

All inference and retraining happen inside the environment (mandatory requirement).

✔ High Accuracy

Achieves 1.00 macro F1 on synthetic, balanced dataset.

✔ Robust Text Cleaning

Handles noisy descriptions, uppercase, punctuation, POS suffixes, etc.

3. Project Architecture
                  ┌─────────────────────────┐
                  │   taxonomy.json          │
                  │ (Admin-configurable)     │
                  └──────────────┬──────────┘
                                 │
                         Load categories
                                 │
┌──────────────────────────┐     ▼       ┌──────────────────────────┐
│ Raw Transaction Text      │ ─────────► │  Text Cleaning + TF-IDF  │
└──────────────────────────┘             └──────────────────────────┘
                                               │
                                               ▼
                                 ┌─────────────────────────┐
                                 │   Logistic Regression    │
                                 │ (trained + retrainable) │
                                 └─────────────────────────┘
                                               │
                           ┌───────────────────┴───────────────────┐
                           ▼                                       ▼
             ┌───────────────────────┐               ┌──────────────────────────┐
             │ Predicted Category     │               │ SHAP Explainability       │
             └───────────────────────┘               └──────────────────────────┘
                           |
                           ▼
        ┌──────────────────────────────────┐
        │ User Feedback (correction input) │
        └──────────────────────────────────┘
                           |
                           ▼
                feedback.csv (audit log)
                           |
                           ▼
                   Retraining Pipeline

4. Installation
Step 1: Clone the Repository
git clone <your_repo_url>
cd <project_folder>

Step 2: Install Dependencies
pip install -r requirements.txt


Recommended versions:

Python 3.9+

scikit-learn 1.3+

streamlit 1.30+

shap 0.44+

5. Running the App Locally

Launch the Streamlit interface:

streamlit run app.py


The browser UI will open automatically.

6. Configurable Taxonomy

Categories are located in:

taxonomy.json


Modify this file to add/remove categories:

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

The app automatically:

Loads new categories

Updates dropdown labels

Maps unknown labels to “Others”

No code change required.

7. How the Feedback Loop Works

User enters a transaction

Model predicts category

User reviews and corrects if needed

Correction is appended to feedback.csv

Example saved row:

description: "UBER TRIP #22193"
predicted: Travel
corrected: Others
corrected_ui: "Taxi"
timestamp: 2024-11-14 23:05:00


Clicking Retrain with Feedback:

Merges corrections into dataset

Retrains model

Saves updated TF-IDF + model .pkl files

This ensures continuous, supervised improvement.

8. Explainability (SHAP)

For each prediction, SHAP reveals:

top contributing keywords

whether they increased or decreased probability

token-level impact

Example output:

feature	shap_value	tfidf_value
starbucks	0.412	0.923
coffee	0.307	0.812
txn	0.021	0.322

This allows transparent inspection of model reasoning.

9. Dataset Documentation

Complete dataset details are in:

DATASET.md


Summary:

Property	Value
Source	Synthetic (merchant pattern generation)
Samples	~1900
Categories	13
Balanced	Yes
Noise simulation	Yes

All synthetic → no privacy risk.

10. Evaluation Results

Full evaluation in:

EVALUATION.md

Macro F1 Score
1.00

Accuracy
1.00

Per-Class F1 Scores

All classes achieved 0.99–1.00.

Confusion Matrix

(Insert confusion matrix image)

11. Responsible AI

Full details in:

RESPONSIBLE_AI.md


Key principles followed:

No sensitive attributes

No external API calls

SHAP transparency

Human-in-loop feedback

No storage of personal identifiers

Balanced dataset prevents category bias

12. File Structure
/project-root
│
├── app.py
├── taxonomy.json
├── synthetic_transactions.csv
├── feedback.csv
├── transaction_classifier_model.pkl
├── tfidf_vectorizer.pkl
│
├── README.md
├── EVALUATION.md
├── DATASET.md
├── RESPONSIBLE_AI.md
│
└── images/
      ├── confusion_matrix.png
      ├── shap_example.png
      └── app_screenshot.png

13. Future Improvements

Train on real-world bank statement samples (when legally possible)

Add neural models (DistilBERT) for more complex phrasing

Add streaming support for real-time classification

Expand noise-robust data augmentation

Build an admin dashboard for retraining and monitoring

14. Conclusion

FinClassify fulfills all GHCI Hackathon requirements:

Requirement	Status
End-to-end categorization	✔
No external APIs	✔
Macro F1 ≥ 0.90	✔ (1.00)
SHAP explainability	✔
Human-in-loop	✔
Configurable taxonomy	✔
Dataset documentation	✔
Responsible AI document	✔
Reproducible pipeline	✔

This system is production-ready for demonstration and robust for future enhancements.