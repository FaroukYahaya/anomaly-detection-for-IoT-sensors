# Anomaly Detection for IoT Sensors

Python implementation of a **multi-detector anomaly-detection pipeline** for streaming IoT sensor data, designed to flag both slow drifts and sudden faults from heterogeneous sensor measurements.

## Detectors in the pipeline

- **EWMA** — exponentially-weighted moving average for slow-drift detection
- **Page–Hinkley** — change-point detector for sudden mean shifts
- **PCA** — principal-component-based residual detector for multivariate dependence

The pipeline also performs **automatic baseline cleaning** (e.g., over a stable Jan–May 2018 reference period) and emits detection events with `fault_type` and `severity` fields.

## Layout

| File | Purpose |
| --- | --- |
| `adm_.py` | Anomaly-detection model (core pipeline) |
| `utils_io.py` | Data I/O helpers |
| `visualization_adm.py` | Plotting utilities |
| `data/` | Example sensor datasets |

## Requirements

Python 3.8+ with NumPy, SciPy, Pandas, scikit-learn, Matplotlib.

## Status

Research code from industrial sensor-network projects. Not a polished package.

## License

MIT. See [LICENSE](LICENSE).
