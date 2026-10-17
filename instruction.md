You are tasked with building an end-to-end tabular ML system to detect chemical compounds from mass spectrometry data.

You are provided with NASA Mars Spectrometry EGA-MS (DrivenData 2022) data containing commercial-instrument and SAM-testbed spectra, with 10 compound-family labels.
Specifically, you have access to:
- Commercial-instrument spectra (m/z x temperature x time) with labels.
- ~10 labelled SAM-testbed samples.
- Preprocessed spectra archives with complete instrument metadata.

Your objective is to train a model to accurately detect these compounds across instruments. The evaluation split contains samples from both instruments (`commercial` and `sam_testbed`). Because mass spectrometry EGA-MS spectra record ion intensities across heating ramps (30°C to 950°C), different chemical compounds release mass fragments at distinct thermal stages (e.g., volatile desorption, organic decomposition, and high-temperature mineral breakdowns). Simple temperature-blind m/z averages fail to disambiguate compounds with shared fragment ions and suffer severe performance degradation under instrument shift. Your modeling approach should capture temperature-resolved spectral features and account for instrument variations to evaluate reliably across both instruments.

### Required Deliverables

1. **Data Preparation & Proxy Validation Artifacts**:
   - Save your declared proxy-validation split to `/app/artifacts/proxy_val_split.csv` containing columns `sample_id`, `split` (with values `train` and `val`), and `instrument_type` (values `commercial` and `sam_testbed`).
     - All `sample_id` values must be drawn from the provided `/app/data/metadata.csv` without synthetic or augmented IDs.
     - The split must be group-disjoint (no sample shared between train and val) and include both `commercial` and `sam_testbed` instruments in each split, covering at least 6 samples in total.
   - Save `/app/artifacts/data_summary.json` summarizing the dataset audit and preprocessing parameters (must include keys `sample_count` matching the integer sample count in `metadata.csv`, `feature_dim` as a positive integer e.g. 300, and `instruments_covered` containing `['commercial', 'sam_testbed']`).

2. **Model Artifact**:
   - Save your trained, loadable model checkpoint to `/app/models/` or `/app/artifacts/` (e.g., `/app/models/best.joblib`, `/app/models/best.pt`, `/app/models/model.pkl`, or `/app/models/model.joblib`).
   - Supported serialization formats include PyTorch (`torch.save` saving an `nn.Module`, `state_dict`, or checkpoint dict), scikit-learn / joblib, or pickle (saving fitted estimators, pipelines, ensembles, or model wrapper classes).
   - The checkpoint must be at least 1 KB and contain trained parameters for the 10 target classes.

3. **Inference Entrypoint**:
   - Write `/app/predict.py` which loads your saved model artifact and accepts command line arguments:
     - `--test_features`: path to a zip file containing test spectra CSVs.
     - `--predictions_output`: path to write the output predictions CSV.
     The predictions CSV must contain `sample_id` and the 10 target compound probability columns (`basalt`, `carbonate`, `chloride`, `iron_oxide`, `oxalate`, `oxychlorine`, `phyllosilicate`, `silicate`, `sulfate`, `sulfide`).
     Predictions must be valid probabilities in [0.0, 1.0] and non-constant across samples (std > 1e-4).
     **Timeout**: `predict.py` must complete inference within 120 seconds.

4. **Metrics**:
   - Write your final evaluation metrics on your proxy-validation split to `/app/artifacts/metrics.json` containing:
     `{"sam_testbed_logloss": <float >= 0.0>, "commercial_logloss": <float >= 0.0>, "macro_ap": <float in [0.0, 1.0]>}`.

5. **Claims**:
   - Write `/app/artifacts/claims.json` where every reported number is traceable to an artifact. It must contain `metrics` (matching values in `metrics.json` within numerical tolerance 1e-3) and a `traceability` object referencing `model_checkpoint` (e.g., `models/best.joblib`) and `validation_split` (e.g., `artifacts/proxy_val_split.csv`).

6. **Report**:
   - Write `/app/artifacts/report.md` (at least 150 words with markdown headings). Describe your model architecture and training setup, including layer dimensions or estimator choices, loss function, and optimizer. Outline your mass spectrometry preprocessing pipeline and feature extraction choices across temperature regimes. Discuss your proxy validation design for handling differences between commercial and SAM-testbed data, and summarize your evaluation results and metrics.

### Data Location
The input data files (`metadata.csv`, `train_labels.csv`, `val_labels.csv`, `train_features.zip`, `val_features.zip`) are provided in `/app/data/`.

### Success Criteria
To successfully pass this task, your model must achieve on the hidden split:
- `sam_testbed_logloss` <= 0.44
- `commercial_logloss` <= 0.46
- `macro_ap` >= 0.80
