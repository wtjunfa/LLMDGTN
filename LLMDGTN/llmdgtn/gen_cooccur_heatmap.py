# -*- coding: utf-8 -*-
"""
图 6 诊断-药物关联热力图（数据驱动，真实可复现）：
用训练集统计「诊断 d 与药物 m 的共现」，以条件概率 P(药物 m | 诊断 d) 作为关联强度，
对患者1 的前两次就诊画热力图。颜色越红 = 该诊断下越常用该药物。
替代原 np.random 模拟 / 梯度归因（后者信号弱、几乎全白）。
"""
import os, sys, io
import dill, numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "mimic-iv")

# ---- 加载数据 ----
data = dill.load(open(os.path.join(data_dir, "records_final.pkl"), "rb"))
voc = dill.load(open(os.path.join(data_dir, "voc_final.pkl"), "rb"))

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]

# ---- 统计训练集诊断-药物共现 ----
diag_count = defaultdict(int)   # 诊断 d 出现的 visit 数
cooccur = defaultdict(int)      # (d, m) 同时出现的 visit 数
for patient in data_train:
    for adm in patient:
        diags = adm[0]
        meds = adm[2]
        for d in diags:
            diag_count[d] += 1
            for m in meds:
                cooccur[(d, m)] += 1

def cond_prob(d, m):
    """P(药物 m | 诊断 d) = 共现次数 / 诊断 d 出现次数"""
    dc = diag_count.get(d, 0)
    return cooccur.get((d, m), 0) / dc if dc > 0 else 0.0

# ---- 定位患者1（按论文 Table 6 诊断代码）----
p1_diag_codes = ['3970','5854','4168','V0481','V4586','2449','3962','9971','53081']
target = set(voc['diag_voc'].word2idx[c] for c in p1_diag_codes if c in voc['diag_voc'].word2idx)
p1 = None
for p in data:
    if len(p) >= 1 and target.issubset(set(p[0][0])):
        p1 = p
        break
assert p1 is not None, "未找到患者1"

diag_voc = voc['diag_voc'].idx2word
med_voc = voc['med_voc'].idx2word

# ---- 画前两次就诊的热力图 ----
n_visits = 2
fig, axes = plt.subplots(1, n_visits, figsize=(7 * n_visits, 6))

for i in range(n_visits):
    visit = p1[i]
    all_meds = set()
    for v in p1[:i + 1]:
        all_meds.update(v[2])
    all_meds = sorted(all_meds)
    all_diags = visit[0]

    # 关联矩阵：P(药物 m | 诊断 d)
    attn = np.zeros((len(all_diags), len(all_meds)))
    for r, d in enumerate(all_diags):
        for c, m in enumerate(all_meds):
            attn[r, c] = cond_prob(d, m)

    ax = axes[i]
    im = ax.imshow(attn, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(all_diags)))
    ax.set_yticklabels([diag_voc.get(d, str(d)) for d in all_diags], fontsize=8)
    ax.set_xticks(range(len(all_meds)))
    ax.set_xticklabels([med_voc.get(m, str(m)) for m in all_meds], rotation=90, fontsize=8)
    ax.set_title(f'({"a" if i == 0 else "b"}) Visit {i + 1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Medications', fontsize=10)
    ax.set_ylabel('Diagnoses', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = os.path.join(base, "saved", "LLMDGTN", "case_study", "Fig8_heatmap.png")
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("已保存诊断-药物关联热力图:", out)
