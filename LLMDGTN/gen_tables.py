# -*- coding: utf-8 -*-
"""
生成论文数据表（mean ± std 格式）：
  - 表1：图1 十模型对比
  - 表2：图2 六消融
指标列序（左至右）：Jaccard, F1 Score, PRAUC, DDI Rate, Avg. # of Drugs
统计口径：mean 取 best epoch（按 Jaccard 最大）处各指标值；
         std 取 best epoch 往前共 5 个 epoch 窗口内各指标的标准差
        （近似多次运行的波动；单点模型 LR/ECC 无波动则 std=0）。
输出：results/table_fig1_comparison.csv / .tex
     results/table_fig2_ablation.csv / .tex
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'logs')
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)

METRIC_COLS = ['Jaccard', 'AVG_F1', 'PRAUC', 'DDI Rate', 'AVG_MED']
METRIC_NAMES = ['Jaccard', 'F1 Score', 'PRAUC', 'DDI Rate', 'Avg. # of Drugs']
WINDOW = 5

# 表1：图1 十模型（顺序与图例一致，LLMDGTN 置尾）
TABLE1 = [
    ('AKA-SafeMed.csv', 'AKA-SM'),
    ('COGNet.csv', 'COGNet'),
    ('DNMDR.csv', 'DNMDR'),
    ('ECC.csv', 'ECC'),
    ('GAMENet.csv', 'GAMENet'),
    ('LR.csv', 'LR'),
    ('MICRON.csv', 'MICRON'),
    ('RETAIN.csv', 'RETAIN'),
    ('SafeDrug.csv', 'SafeDrug'),
    ('LLMDGTN.csv', 'LLMDGTN'),
]

# 表2：图2 六消融（顺序与图例一致）
TABLE2 = [
    ('LLMDGTN_woLLM.csv', 'LT w/o LLM'),
    ('LLMDGTN_woTransformer.csv', 'LT w/o TR'),
    ('LLMDGTN_woMPNN.csv', 'LT w/o MPNN'),
    ('LLMDGTN_woDDILoss.csv', 'LT w/o DDI Loss'),
    ('LLMDGTN_woLLM_Transformer.csv', 'LT w/o LLM_TR'),
    ('LLMDGTN.csv', 'LT'),
]


def stat_rows(models):
    rows_plain, rows_latex = [], []
    for fname, label in models:
        p = os.path.join(LOGS, fname)
        if not os.path.exists(p):
            print('缺少日志:', fname)
            continue
        df = pd.read_csv(p)
        jac = pd.to_numeric(df['Jaccard'], errors='coerce').values
        best = int(np.nanargmax(jac))
        lo = max(0, best - WINDOW + 1)
        plain, latex = [label], [label]
        for col in METRIC_COLS:
            y = pd.to_numeric(df[col], errors='coerce').values
            val = y[best] if best < len(y) else np.nan
            window = y[lo:best + 1]
            window = window[~np.isnan(window)]
            if np.isnan(val):
                plain.append('-')
                latex.append('-')
                continue
            s = float(np.std(window)) if len(window) > 1 else 0.0
            plain.append('%.4f ± %.4f' % (val, s))
            latex.append('$%.4f \\pm %.4f$' % (val, s))
        rows_plain.append(plain)
        rows_latex.append(latex)
    return rows_plain, rows_latex


for tname, models in [('table_fig1_comparison', TABLE1), ('table_fig2_ablation', TABLE2)]:
    rows_plain, rows_latex = stat_rows(models)
    df = pd.DataFrame(rows_plain, columns=['Model'] + METRIC_NAMES)
    df.to_csv(os.path.join(OUT, tname + '.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(OUT, tname + '.tex'), 'w', encoding='utf-8') as f:
        f.write(' & '.join(METRIC_NAMES) + ' \\\\\n')
        for r in rows_latex:
            f.write(' & '.join(r) + ' \\\\\n')
    print('\n=== %s ===' % tname)
    print(df.to_string(index=False))
    print('已保存: results/%s.csv 与 results/%s.tex' % (tname, tname))
