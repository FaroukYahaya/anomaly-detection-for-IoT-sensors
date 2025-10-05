#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Farouk Yahaya
Contact: faroya2011@gmail.com
Date: 2025-10-04

Title: Anomaly Detection Model (ADM)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from river.drift import PageHinkley

from utils_io import setup_logger
from visualization_adm import plot_detection_summary

BASELINE_CSV = Path("data/ChillerPlant_extracted.csv")
FAULTY_CSV   = Path("data/ChillerPlant_chiller_bias_2_extracted.csv")
TIMESTAMP_COL = "Datetime"
CHW_COL = "CWL_SEC_SW_TEMP"
TZ = "Europe/Paris"
WORK_START, WORK_END = "07:00", "18:00"
PCA_FEATURES = ("CWL_SEC_SW_TEMP","CWL_SEC_RW_TEMP","CWL_SEC_CW_FLOW","CWL_SEC_DP","CWL_SEC_LOAD")
OUT_DIR = Path("adm_outputs")

def _load(csv: Path, tcol: str, cols: list[str], tz: str) -> pd.DataFrame:
    use = [tcol] + cols
    df = pd.read_csv(csv, usecols=lambda c: c in use)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        df = df[~df.index.isna()]
    else:
        df.index = df.index.tz_convert(tz)
    hhmm = df.index.strftime("%H:%M")
    df = df.loc[(hhmm >= WORK_START) & (hhmm < WORK_END)]
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return df

def _tod(series: pd.Series) -> tuple[pd.Series, float]:
    s = series.to_frame("y")
    s["hhmm"] = s.index.strftime("%H:%M")
    mu = s.groupby("hhmm")["y"].mean()
    s["mu"] = s["hhmm"].map(mu); s["r"] = s["y"] - s["mu"]
    return mu, float(s["r"].std(ddof=1))

def detect_fault_events() -> dict[str,str|None]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUT_DIR/"logs"/"adm_run.log")
    base_u = _load(BASELINE_CSV, TIMESTAMP_COL, [CHW_COL], TZ)
    test_u = _load(FAULTY_CSV, TIMESTAMP_COL, [CHW_COL], TZ)
    mu, sigma = _tod(base_u[CHW_COL])
    ts = test_u.copy()
    ts["hhmm"] = ts.index.strftime("%H:%M"); ts["mu"] = ts["hhmm"].map(mu); ts["r"] = ts[CHW_COL] - ts["mu"]
    # EWMA
    alpha, L = 0.2, 3.0
    Z = ts["r"].ewm(alpha=alpha, adjust=False).mean()
    n = np.arange(1, len(ts)+1)
    factor = np.sqrt(alpha/(2-alpha) * (1 - (1-alpha)**(2*n)))
    UCL = L*sigma*factor; LCL = -UCL
    ewma_idx = np.where((Z.values>UCL) | (Z.values<LCL))[0]
    t_ewma = ts.index[ewma_idx[0]] if len(ewma_idx)>0 else None
    logger.info(f"EWMA detection: {t_ewma}")
    # Page–Hinkley
    ph = PageHinkley(min_instances=30, delta=0.0, threshold=50.0, alpha=1.0)
    t_ph = None
    for t, x in ts["r"].items():
        ph.update(float(x))
        if ph.drift_detected:
            t_ph = t; break
    logger.info(f"Page–Hinkley detection: {t_ph}")
    # PCA
    base_m = _load(BASELINE_CSV, TIMESTAMP_COL, list(PCA_FEATURES), TZ)
    test_m = _load(FAULTY_CSV, TIMESTAMP_COL, list(PCA_FEATURES), TZ)
    scaler = StandardScaler()
    Xb = scaler.fit_transform(base_m.values); Xt = scaler.transform(test_m.values)
    pca = PCA(svd_solver="full", random_state=0).fit(Xb)
    lam = pca.explained_variance_
    Zb = pca.transform(Xb); Zt = pca.transform(Xt)
    T2_b = np.sum((Zb**2)/lam, axis=1); T2_t = np.sum((Zt**2)/lam, axis=1)
    Q_b = np.sum((Xb - pca.inverse_transform(Zb))**2, axis=1)
    Q_t = np.sum((Xt - pca.inverse_transform(Zt))**2, axis=1)
    thr_T2 = float(np.quantile(T2_b, 0.99)); thr_Q = float(np.quantile(Q_b, 0.99))
    idx_T2 = np.where(T2_t>thr_T2)[0]; idx_Q = np.where(Q_t>thr_Q)[0]
    if len(idx_T2)>0 or len(idx_Q)>0:
        i = min(idx_T2[0] if len(idx_T2)>0 else len(T2_t), idx_Q[0] if len(idx_Q)>0 else len(Q_t))
        t_pca = test_m.index[i]
    else:
        t_pca = None
    logger.info(f"PCA detection: {t_pca}")
    # Save CSV
    out = pd.DataFrame([
        {"Detector":"EWMA","DetectionTime": str(t_ewma) if t_ewma else "No detection"},
        {"Detector":"PageHinkley","DetectionTime": str(t_ph) if t_ph else "No detection"},
        {"Detector":"PCA","DetectionTime": str(t_pca) if t_pca else "No detection"},
    ])
    out.to_csv(OUT_DIR/"adm_results.csv", index=False)
    # Plot
    plot_detection_summary(test_u, CHW_COL, {"EWMA":t_ewma, "Page-Hinkley":t_ph, "PCA":t_pca}, OUT_DIR/"adm_detection.png")
    return {"EWMA":t_ewma, "PageHinkley":t_ph, "PCA":t_pca}

if __name__ == "__main__":
    detect_fault_events()
