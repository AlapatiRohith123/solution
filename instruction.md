You are tasked with building an end-to-end tabular ML system to detect chemical compounds from mass spectrometry data.

You are provided with NASA Mars Spectrometry EGA-MS (DrivenData 2022) data containing commercial-instrument and SAM-testbed spectra, with 10 compound-family labels.
Specifically, you have access to:
- Commercial-instrument spectra (m/z x temperature x time) with labels.
- ~10 labelled SAM-testbed samples.
- Unlabelled SAM-testbed spectra.

Your objective is to train a model to accurately detect these compounds. We will evaluate your solution on a hidden verifier-owned split containing the remaining SAM-testbed samples and commercial competition-test samples.

Note on Evaluation Data Shift:
The hidden evaluation data features a distribution shift: files are renamed to opaque IDs, temperature ramps are resampled, and intensities are scaled by a secret per-instrument factor. Ensure your proxy validation design and modeling approach specifically accounts for this per-instrument shift.

### Required Deliverables

1. **Data Preparation & Proxy Validation Artifacts**:
   - Save your declared proxy-validation split to `/app/artifacts/proxy_val_split.csv` containing columns `sample_id` and `split` (with values `train` and `val`). To ensure valid evaluation under distribution shift, your split must be group-disjoint (no sample shared between train and val) and include instrument assignments reflecting the shift.
   - Save `/app/artifacts/data_summary.json` summarizing the dataset audit and preprocessing parameters (must include keys `sample_count`, `feature_dim`, and `instruments_covered`).

2. **Model Artifact**:
   - Save your trained, loadable model checkpoint to `/app/models/` (e.g., `/app/models/best.pt`).

3. **Inference Entrypoint**:
   - Write `/app/predict.py` which loads your saved model artifact and accepts command line arguments:
     - `--test_features`: path to a zip file containing test spectra CSVs.
     - `--predictions_output`: path to write the output predictions CSV.
     The predictions CSV must contain `sample_id` and the 10 target compound probability columns (`basalt`, `carbonate`, `chloride`, `iron_oxide`, `oxalate`, `oxychlorine`, `phyllosilicate`, `silicate`, `sulfate`, `sulfide`).

4. **Metrics**:
   - Write your final evaluation metrics on your proxy-validation split to `/app/artifacts/metrics.json` containing:
     `{"sam_testbed_logloss": <value>, "commercial_logloss": <value>, "macro_ap": <value>}`.

5. **Claims**:
   - Write `/app/artifacts/claims.json` where every reported number is traceable to an artifact. It must contain `metrics` (matching values in `metrics.json` within numerical tolerance) and a `traceability` object referencing `model_checkpoint` and `validation_split`.

6. **Report**:
   - Write `/app/artifacts/report.md` (at least 150 words with markdown headings) covering:
     - Model architecture and training configuration.
     - Mass spectrometry preprocessing and feature extraction (m/z binning, intensity normalization).
     - Instrument distribution shift handling and group-disjoint proxy validation strategy.
     - Performance evaluation and discussion of results.

### Data Location
The input data files (`metadata.csv`, `train_labels.csv`, `val_labels.csv`, `train_features.zip`, `val_features.zip`) are provided in `/app/data/`.

### Success Criteria
To successfully pass this task, your model must achieve on the hidden split:
- `sam_testbed_logloss` <= 0.66
- `commercial_logloss` <= 0.65
- `macro_ap` >= 0.40
