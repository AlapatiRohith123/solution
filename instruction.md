You are tasked with building an end-to-end tabular ML system to detect chemical compounds from mass spectrometry data.

You are provided with NASA Mars Spectrometry EGA-MS (DrivenData 2022) data containing commercial-instrument and SAM-testbed spectra, with 10 compound-family labels.
Specifically, you have access to:
- Commercial-instrument spectra (m/z x temperature x time) with labels.
- ~10 labelled SAM-testbed samples.
- Unlabelled SAM-testbed spectra.

Your objective is to train a model to accurately detect these compounds. We will evaluate your solution on a hidden verifier-owned split containing the remaining SAM-testbed samples and commercial competition-test samples.

Deliverables:
- A loadable model.
- An inference entry point (`predict.py` or similar) that we can run.
- Write your final evaluation metrics to `artifacts/metrics.json` as a JSON object with the primary metrics on a proxy-validation split (e.g. `{"sam_testbed_logloss": <value>, "commercial_logloss": <value>, "macro_ap": <value>}`).
- `claims.json` where every reported number is traceable to an artifact.
- A short report summarizing your approach.

Note on Evaluation Data Shift:
The hidden evaluation data features a distribution shift: files are renamed to opaque IDs, temperature ramps are resampled, and intensities are scaled by a secret per-instrument factor. Ensure your proxy validation and modeling approach accounts for this per-instrument shift.

### Data and Deliverables
The input data files (including `dataset.csv` and NPZ files) are provided in the relative directory `data/` (e.g. `data/dataset.csv`).
Your code must output its final metrics to `artifacts/metrics.json`, claims to `artifacts/claims.json`, and write a `predict.py` in the working directory.

### Success Criteria
To successfully pass this task, your model must achieve:
- `sam_testbed_logloss` <= 0.5
- `commercial_logloss` <= 0.5
- `macro_ap` >= 0.15
