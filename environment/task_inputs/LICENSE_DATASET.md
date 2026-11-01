# OPSSAT-AD Telemetry Anomaly Detection Benchmark License & Attribution

This benchmark dataset is derived from the European Space Agency (ESA) OPS-SAT spacecraft telemetry benchmark:

**OPS-SAT Telemetry Anomaly Detection Benchmark (OPSSAT-AD)**  
European Space Agency (ESA) / OPS-SAT Mission Team  
Zenodo Record: [10.5281/zenodo.12588359](https://doi.org/10.5281/zenodo.12588359)  
Preprint: Ruszczak, B., Kotowski, K., Nalepa, J., & Evans, D. (2024). *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*. arXiv:2407.04730.

The source dataset is distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**:  
https://creativecommons.org/licenses/by/4.0/

## Dataset Provenance & Transformations
The telemetry sequences provided in `dataset.csv` represent curated univariate time-series sequences across multiple telemetry channels acquired on board the OPS-SAT CubeSat in orbit. Telemetry channels cover key spacecraft subsystems:
- Electrical Power System (EPS): Solar array currents and battery voltages (`channel_01`, `channel_02`)
- Attitude Determination and Control System (ADCS): Gyro reaction wheel rates (`channel_03`, `channel_08`)
- Thermal Control System (TCS): Radiator plate and battery temperatures (`channel_04`, `channel_06`)
- On-Board Computer (OBC): Telemetry buffer utilization and stream packet counters (`channel_07`, `channel_09`)
- Communications / Payload Interface: Downlink bus telemetry (`channel_05`)

### Transformations for Benchmark Integrity
- Continuous telemetry fragments have been resampled to standardized 128-step normalized sequences (`seq_000` through `seq_127`).
- Timestamps and internal segment indices have been re-encoded with opaque segment identifiers (`frag_XXXX`) to prevent label lookup against raw public tables.
- The training split contains 1,740 sequences from 6 visible telemetry channels (`channel_01` through `channel_06`).
- Approximately 20% (346 sequences) carry verified ground-truth anomaly annotations (`anomaly` is 0 or 1), while the remaining ~80% (1,394 sequences) are unlabelled (`anomaly` is -1).
- Evaluation test sets evaluate anomaly detection generalization under channel distribution shift on visible channels (`in_domain_test.csv`) and held-out channels (`hidden_test.csv`).
