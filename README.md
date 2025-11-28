# 🚢 Titanic — Baseline Pipeline & Submission  
### Simple, well-structured Kaggle baseline using RandomForest (functions + `main()`)

An end-to-end baseline pipeline for the Kaggle **Titanic: Machine Learning from Disaster** competition.  
This repo contains a clean, function-based script that loads data, performs light feature engineering, trains a RandomForest baseline, evaluates on a validation split, and writes a Kaggle-ready submission.

---

## 📘 Competition  
**Kaggle:** https://www.kaggle.com/competitions/titanic

**Expected files:**  
- `train.csv` → training data with `Survived` target  
- `test.csv` → test data to predict `Survived`  
*(Datasets not included in this repo — download from Kaggle and place in project root.)*

---

## ⚙️ What this Project Does (overview)

1. Loads `train.csv` and `test.csv`.  
2. Preprocesses data:
   - Keeps selected useful columns.  
   - Fills missing `Age`, `Fare` (median) and `Embarked` (mode).  
   - Creates `FamilySize = SibSp + Parch + 1`.  
   - Extracts `Title` from `Name` and groups rare titles.  
   - One-hot encodes categorical features (`Sex`, `Embarked`, `Title`).  
3. Splits processed train data into train/validation (stratified).  
4. Trains a RandomForest baseline and prints validation metrics (accuracy + classification report).  
5. Trains final model on full training data and generates `submission_baseline_rf.csv`.

---

## 🧠 Model & Feature Notes

- **Baseline model:** `RandomForestClassifier`
  - Baseline training: `n_estimators=300, max_depth=6, min_samples_leaf=4`
  - Final training on full data: `n_estimators=400, max_depth=7, min_samples_leaf=3`
- **Key engineered features:**
  - `FamilySize`
  - `Title` extracted and simplified from `Name`
  - One-hot encoded `Sex`, `Embarked`, `Title`
- This pipeline is intentionally simple and reproducible — a strong baseline before adding more advanced features (age bins, cabin processing, ticket grouping, target encoding, stacking).

---

## 🚀 Quick Start — Run locally

Place `train.csv` and `test.csv` in the repo root, then run:

```bash
python titanic_pipeline.py
```
What happens:

Displays basic shapes and a sample of processed rows.

Runs train/validation split and trains baseline model.

Prints validation accuracy + classification report.

Trains final model on full training data and saves:
submission_baseline_rf.csv (columns: PassengerId, Survived).

📁 Repository structure

├── titanic_pipeline.py          # main script (functions + main)
├── submission_baseline_rf.csv   # generated submission (after running)
├── train.csv                    # (not included)
├── test.csv                     # (not included)
├── requirements.txt
└── README.md
✅ Tips to Improve Performance
Better Age imputation by Title or Pclass + Title groups.

Extract features from Ticket or Cabin (deck).

Create categorical bins for Fare and Age.

Use target encoding or cross-validated mean encoding for high-cardinality features.

Try LightGBM / XGBoost / stacking for stronger leaderboard results.

👤 Author
Puneet Poddar
Kaggle: https://www.kaggle.com/puneet2769

📌 License / Attribution
Use freely for learning and experimentation. When publishing results derived from this pipeline, credit the original dataset source (Kaggle).
