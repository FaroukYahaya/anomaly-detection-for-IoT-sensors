# Anomaly Detection for IoT Sensors

    [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()

    A modular Python framework implementing EWMA, Page-Hinkley, and PCA-based anomaly detection for cleanroom HVAC sensor data.

    ## Features
    - Multi-detector pipeline: **EWMA**, **Page–Hinkley**, **PCA**
- Automatic baseline cleaning (e.g., Jan–May 2018 stable period)
- Detection events with fields: *fault_type*, *severity*, *timestamp*, *latency*
- Exports CSV/TeX; IEEE-grade plots with **pgfplots**
- Hooks to route anomalies to the **SCM** (calibration) module

    ## Quickstart
    ```bash
    git clone https://github.com/FaroukYahaya/anomaly-detection-for-iot-sensors.git
    cd anomaly-detection-for-iot-sensors

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    # source .venv/bin/activate  # macOS/Linux

    pip install -r requirements.txt
    ```

    ## Repository Layout (suggested)
    ```
    .
    ├── src/                # core modules
    ├── data/               # (keep small; avoid >100MB)
    ├── docs/               # figures, LaTeX, notes
    ├── tests/              # unit tests
    ├── README.md
    ├── .gitignore
    └── requirements.txt
    ```

    ## Author
    Farouk Yahaya

    ## License
    MIT
