You are tasked with building an end-to-end tabular ML system to detect chemical compounds from mass spectrometry data.

You are provided with NASA Mars Spectrometry EGA-MS (DrivenData 2022) data containing commercial-instrument and SAM-testbed spectra, with 10 compound-family labels.
Specifically, you have access to:
- Commercial-instrument spectra (m/z x temperature x time) with labels.
- ~10 labelled SAM-testbed samples.
- Preprocessed spectra archives with complete instrument metadata.

Your objective is to train a model to accurately detect these compounds across instruments. We will evaluate your solution on a hidden verifier-owned split containing SAM-testbed samples and commercial test samples.

Note on Evaluation Data Shift:
The hidden evaluation data features a distribution shift: files are renamed to opaque IDs, temperature ramps are resampled, and intensities are scaled by a secret per-instrument factor. Ensure your proxy validation design and modeling approach specifically accounts for this per-instrument shift via intensity normalization and instrument-aware validation.

### Required Deliverables

1. **Data Preparation & Proxy Validation Artifacts**:
   - Save your declared proxy-validation split to `/app/artifacts/proxy_val_split.csv` containing columns `sample_id`, `split` (with values `train` and `val`), and `instrument_type` (values `commercial` and `sam_testbed`).
     - **Constraint**: All `sample_id` values must be drawn from the provided `/app/data/metadata.csv` (do not add synthetic or augmented sample IDs).
     - **Distribution Shift**: The split must be strictly group-disjoint (no sample shared between train and val) and must include both `commercial` and `sam_testbed` instruments in both splits (`train` and `val`).
   - Save `/app/artifacts/data_summary.json` summarizing the dataset audit and preprocessing parameters (must include keys `sample_count` matching the total sample count in `metadata.csv`, `feature_dim` e.g. 100, and `instruments_covered` containing `['commercial', 'sam_testbed']`).

2. **Model Artifact**:
   - Save your trained, loadable model checkpoint to `/app/models/` (e.g., `/app/models/best.pt`). You may serialize using `torch.save(...)` for PyTorch or `joblib`/`pickle`.
   - The checkpoint must be at least 1 KB and contain substantive trained parameters for all 10 target classes.

3. **Inference Entrypoint**:
   - Write `/app/predict.py` which loads your saved model artifact and accepts command line arguments:
     - `--test_features`: path to a zip file containing test spectra CSVs.
     - `--predictions_output`: path to write the output predictions CSV.
     The predictions CSV must contain `sample_id` and the 10 target compound probability columns (`basalt`, `carbonate`, `chloride`, `iron_oxide`, `oxalate`, `oxychlorine`, `phyllosilicate`, `silicate`, `sulfate`, `sulfide`).
     Predictions must be valid probabilities in [0.0, 1.0] and non-constant across samples (std > 1e-4).

4. **Metrics**:
   - Write your final evaluation metrics on your proxy-validation split to `/app/artifacts/metrics.json` containing:
     `{"sam_testbed_logloss": <value>, "commercial_logloss": <value>, "macro_ap": <value>}`.

5. **Claims**:
   - Write `/app/artifacts/claims.json` where every reported number is traceable to an artifact. It must contain `metrics` (matching values in `metrics.json` within numerical tolerance 1e-3) and a `traceability` object referencing `model_checkpoint` (e.g., `models/best.pt`) and `validation_split` (e.g., `artifacts/proxy_val_split.csv`).

6. **Report**:
   - Write `/app/artifacts/report.md` (at least 250 words with markdown headings) covering:
     - Model architecture, training configuration, loss, and regularization.
     - Mass spectrometry preprocessing and feature extraction (m/z binning, intensity normalization).
     - Instrument distribution shift handling and group-disjoint, instrument-stratified proxy validation strategy.
     - Performance evaluation and discussion of results.

### Data Location
The input data files (`metadata.csv`, `train_labels.csv`, `val_labels.csv`, `train_features.zip`, `val_features.zip`) are provided in `/app/data/`.

### Success Criteria
To successfully pass this task, your model must achieve on the hidden split:
- `sam_testbed_logloss` <= 0.64
- `commercial_logloss` <= 0.63
- `macro_ap` >= 0.50
