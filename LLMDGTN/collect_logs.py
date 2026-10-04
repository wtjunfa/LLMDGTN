# -*- coding: utf-8 -*-
"""
收集各模型的训练日志，统一生成 logs/ 下的 CSV（供画图脚本读取）。

用法：跑完所有模型后，在本目录执行
    python collect_logs.py

它会：
1. 把 llmdgtn/saved/{model}/training_summary.csv 复制为 logs/{model}.csv
2. 把 baselines/baselines/saved/{model}/history.pkl 转为 logs/{model}.csv
"""
import os, sys, io, shutil, glob
import dill
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'logs')
os.makedirs(LOGS, exist_ok=True)

# 以下日志无重跑代码（沿用旧值）：自动备份/恢复，避免被清空丢失
PROTECTED = {
    'AKA-SafeMed.csv': 'AKA-SafeMed 无公开代码（沿用旧值）',
    'results.csv': '图3 就诊次数（已用 best 模型评估，备份防误删）',
}
for fname, reason in PROTECTED.items():
    log_f = os.path.join(LOGS, fname)
    bak_f = os.path.join(ROOT, 'data', fname.replace('.csv', '_old.csv'))
    if os.path.exists(log_f) and os.path.getsize(log_f) > 0:
        shutil.copy2(log_f, bak_f)
        print('已备份 %s -> data/%s（%s）' % (fname, os.path.basename(bak_f), reason))
    elif os.path.exists(bak_f) and os.path.getsize(bak_f) > 0:
        shutil.copy2(bak_f, log_f)
        print('已从备份恢复 %s（%s）' % (fname, reason))

# 1) LLMDGTN 及消融变体：training_summary.csv -> logs/*.csv
summary_glob = os.path.join(ROOT, 'llmdgtn', 'saved', '*', 'training_summary.csv')
for f in sorted(glob.glob(summary_glob)):
    model = os.path.basename(os.path.dirname(f))
    shutil.copy2(f, os.path.join(LOGS, model + '.csv'))
    print('LLMDGTN类 -> logs/%s.csv' % model)

# 优化后的最终 LLMDGTN 主模型结果（覆盖 logs/LLMDGTN.csv，供图1 使用）
BEST_SEED = 'LLMDGTN_opt'
best_summary = os.path.join(ROOT, 'llmdgtn', 'saved', BEST_SEED, 'training_summary.csv')
if os.path.exists(best_summary):
    shutil.copy2(best_summary, os.path.join(LOGS, 'LLMDGTN.csv'))
    print('最终模型 %s -> logs/LLMDGTN.csv（覆盖）' % BEST_SEED)

# 2) 基线：history.pkl -> logs/*.csv
hist_globs = [
    os.path.join(ROOT, 'baselines', 'baselines', 'saved', '*', 'history*.pkl'),
    os.path.join(ROOT, 'baselines', 'dnmdr', 'saved', '*', 'history*.pkl'),
]
for hg in hist_globs:
    for f in sorted(glob.glob(hg)):
        model = os.path.basename(os.path.dirname(f))
        hist = dill.load(open(f, 'rb'))
        jac = hist.get('jaccard', hist.get('ja', []))
        n = len(jac)
        df = pd.DataFrame({
            'DDI Rate': hist.get('ddi_rate', [0] * n),
            'Jaccard': jac,
            'PRAUC': hist.get('prauc', [0] * n),
            'AVG_F1': hist.get('avg_f1', [0] * n),
            'AVG_MED': hist.get('med', [0] * n),
            'best_epoch': hist.get('best_epoch', [0] * n),
        })
        df.to_csv(os.path.join(LOGS, model + '.csv'), index=False)
        print('基线 -> logs/%s.csv (%d epochs)' % (model, len(df)))

print('\n日志已收集到:', LOGS)
