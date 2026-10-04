# -*- coding: utf-8 -*-
"""
图 3：不同模型性能随就诊次数（number of visits）的变化
数据来源：results.csv（列：Model, Visits, Patients, DDI_Rate, Jaccard, PRAUC, AVG_F1, AVG_Med）

用法：
    1. 把多模型就诊次数实验的 results.csv 放到 logs/ 目录
    2. python plot_fig3_visits.py
    输出：Fig3_visits.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

# ============ 可修改配置 ============
MODEL_ORDER = ['LR', 'GAMENet', 'SafeDrug', 'LLMDGTN']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'logs', 'results.csv')
OUT_DIR = os.path.join(ROOT, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [
    ('Jaccard', 'Jaccard', 'Jaccard Score'),
    ('AVG_F1', 'F1 Score', 'F1 Score'),
    ('PRAUC', 'PRAUC', 'PRAUC'),
    ('DDI_Rate', 'DDI Rate', 'DDI Rate'),
]

COLORS = {'LR': '#1f77b4', 'GAMENet': '#ff7f0e', 'SafeDrug': '#2ca02c', 'LLMDGTN': '#d62728'}
MARKERS = {'LR': 'o', 'GAMENet': '^', 'SafeDrug': 's', 'LLMDGTN': 'D'}


def main():
    if not os.path.exists(CSV_PATH):
        print('未找到 results.csv，请放到:', CSV_PATH)
        return
    df = pd.read_csv(CSV_PATH)

    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.6))
    fig.patch.set_facecolor('white')

    for ax, (col, title, ylab) in zip(axes, METRICS):
        for m in MODEL_ORDER:
            sub = df[df['Model'] == m].sort_values('Visits')
            if sub.empty:
                continue
            x = sub['Visits'].values
            y = pd.to_numeric(sub[col], errors='coerce').values
            ax.plot(x, y, color=COLORS.get(m), marker=MARKERS.get(m, 'o'),
                    markersize=5, linewidth=1.8, label=m, alpha=0.9,
                    markerfacecolor=COLORS.get(m), markeredgecolor='white', markeredgewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Visits', fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='lower right', frameon=True, fancybox=False,
                      edgecolor='black', framealpha=1.0)

    plt.tight_layout(pad=1.2)
    out = os.path.join(OUT_DIR, 'Fig3_visits.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('图3 已保存:', out)


if __name__ == '__main__':
    main()
