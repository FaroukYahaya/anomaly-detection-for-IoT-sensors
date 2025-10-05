#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Farouk Yahaya
Contact: faroya2011@gmail.com
Date: 2025-10-04

Title: ADM Visualization
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Disable LaTeX and set simple fonts
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'


def plot_detection_summary(series_df, chw_col, detections: dict, out_png):
    # Define distinct colors and markers for each detector
    detector_styles = {
        'EWMA': {'color': 'red', 'linestyle': '--', 'marker': 'o', 'markersize': 10},
        'Page-Hinkley': {'color': 'green', 'linestyle': '-.', 'marker': 's', 'markersize': 9},
        'PCA': {'color': 'orange', 'linestyle': ':', 'marker': '^', 'markersize': 10}
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series_df.index, series_df[chw_col], label="CHW-S", linewidth=1.5, color='blue', alpha=0.7)

    for det_name, det_time in detections.items():
        if det_time not in (None, "No detection"):
            style = detector_styles.get(det_name, {'color': 'black', 'linestyle': '-', 'marker': 'x', 'markersize': 8})

            # Plot vertical line
            ax.axvline(det_time, linestyle=style['linestyle'], color=style['color'],
                       linewidth=2, alpha=0.8, label=f"{det_name}: {det_time}")

            # Add marker at detection point
            if det_time in series_df.index:
                y_val = series_df.loc[det_time, chw_col]
            else:
                idx = series_df.index.get_indexer([det_time], method='nearest')[0]
                y_val = series_df.iloc[idx][chw_col]

            ax.plot(det_time, y_val, marker=style['marker'], color=style['color'],
                    markersize=style['markersize'], markeredgewidth=1.5,
                    markeredgecolor='black', markerfacecolor=style['color'])

    ax.set_title("ADM detections (timeline)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temp (C)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
