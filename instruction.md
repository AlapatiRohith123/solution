You are tasked with building an end-to-end tabular ML system for the Mars Evolved-Gas Spectra: Multi-Label Compound Detection From Commercial Instruments to the SAM Testbed task.

You are provided with NASA Mars Spectrometry EGA-MS (DrivenData 2022) data containing commercial-instrument and SAM-testbed spectra, with 10 compound-family labels.
Specifically, you have access to:
- Commercial-instrument spectra (m/z x temperature x time) with labels.
- ~10 labelled SAM-testbed samples.
- Unlabelled SAM-testbed spectra.

Your objective is to train a model to accurately detect these compounds. We will evaluate your solution on a hidden verifier-owned split containing the remaining SAM-testbed samples and commercial competition-test samples.

Deliverables:
- A loadable model.
- An inference entry point (`predict.py` or similar) that we can run.
- Write your final evaluation metrics to `/app/artifacts/metrics.json` as a JSON object with the primary metrics on a proxy-validation split (e.g. `{"sam_testbed_logloss": <value>, "commercial_logloss": <value>, "macro_ap": <value>}`).
- `claims.json` where every reported number is traceable to an artifact.
- A short report summarizing your approach.

Note on Evaluation Data Shift:
The hidden evaluation data features a distribution shift: files are renamed to opaque IDs, temperature ramps are resampled, and intensities are scaled by a secret per-instrument factor. Ensure your proxy validation and modeling approach accounts for this per-instrument shift.
