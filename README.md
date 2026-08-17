# Gastric Cancer Postoperative Survival Calculator

Academic Streamlit app for the ExtraSurvivalTrees model selected by nested cross-validation (mean outer-fold Uno C-index 0.790). For research demonstration only. Not a medical device and not for clinical decision-making.

## Local run

```powershell
streamlit run app.py
```

Python 3.12. Dependencies: `requirements.txt`.

## Streamlit Community Cloud

1. Create a GitHub repository whose **root** is this folder (not the parent analysis project).
2. Push only de-identified files. Patient IDs and raw cohort tables are gitignored.
3. Open [https://share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app**.
4. Select this repository, branch `main`, and main file `app.py`.
5. Deploy. The public URL will look like `https://<app-name>.streamlit.app`.

The model file is about 78 MB, below GitHub’s 100 MB file limit, so Git LFS is not required.

## What the app does

- Single-patient input of 10 final variables (`IER5L_Expersion`, radiomics score RS, clinical and laboratory variables)
- Risk score, high/low-risk group, 1-/3-/5-year calibrated event probabilities
- Individual predicted survival curve overlaid on cohort KM curves

## Required disclosures

- RS must be precomputed; the app cannot extract features from CT
- Training pTNM values are 2 and 3 only
- The risk cutoff is the training-cohort median, not a clinical guideline
- Calibrated probabilities are more optimistic than strict out-of-fold estimates
- No user inputs are saved

## Public artifact policy

Published files contain the frozen model, preprocessor, Cox calibrator, aggregated KM coordinates, and anonymized examples. They do **not** contain patient IDs or individual outcome tables.
