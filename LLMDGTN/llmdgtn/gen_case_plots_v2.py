# -*- coding: utf-8 -*-
"""
改进版推荐案例图（图7/8）：
  - 用 logits 展示（sigmoid 前的模型真实分数，有区分度，避免概率饱和成 1.0）
  - 红=实际处方，蓝=推荐但未处方，带图例
  - 清晰展示 top-30 推荐
"""
import os, sys, io
import dill, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base = os.path.dirname(os.path.abspath(__file__))
case_dir = os.path.join(base, 'saved', 'LLMDGTN', 'case_study')
case = dill.load(open(os.path.join(case_dir, 'case_study_data.pkl'), 'rb'))
med_voc = case['voc']['med']

def sigmoid_to_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def plot_recommendation(visit, fig_name, title):
    top_meds = visit['top_meds_idx'][:30]
    top_probs = np.array(visit['top_meds_probs'][:30])
    actual = set(visit['actual_meds_idx'])

    # 用 logits 展示（有区分度）
    logits = sigmoid_to_logit(top_probs)
    order = np.argsort(logits)  # 升序
    meds_sorted = [top_meds[i] for i in order]
    logits_sorted = logits[order]
    labels = [med_voc.get(m, med_voc.get(str(m), f'idx_{m}')) for m in meds_sorted]
    colors = ['#C0392B' if m in actual else '#2E86C1' for m in meds_sorted]

    fig, ax = plt.subplots(figsize=(7, 8.5))
    y_pos = np.arange(len(meds_sorted))
    ax.barh(y_pos, logits_sorted, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Prediction Logit (higher = more confident)', fontsize=11)
    ax.set_ylabel('Drug ATC Code', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # 红蓝图例
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#C0392B', label='Actually prescribed'),
                       Patch(color='#2E86C1', label='Recommended but not prescribed')],
              loc='lower right', fontsize=9, frameon=True, fancybox=False, edgecolor='black')

    # 标注 logit 值
    for bar, l in zip(ax.patches, logits_sorted):
        ax.text(l + 0.1, bar.get_y() + bar.get_height() / 2, f'{l:.1f}',
                va='center', fontsize=6.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(case_dir, fig_name)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('已保存:', out)

# 图7：患者1 visit3（第三次就诊）
p1 = case['patient_3visits']['visits']
if len(p1) >= 3:
    plot_recommendation(p1[2], 'Fig9_recommend_patient1.png',
                        'Top-30 Recommended Medications for Patient 1 at Visit 3')

# 图8：患者2 visit1
p2 = case['patient_1visit']['visits']
plot_recommendation(p2[0], 'Fig10_recommend_patient2.png',
                    'Top-30 Recommended Medications for Patient 2 at Visit 1')
