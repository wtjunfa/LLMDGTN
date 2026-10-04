# -*- coding: utf-8 -*-
"""
图 6 真实注意力热力图（方案 A）：
用优化后 LLMDGTN 学到的「诊断嵌入 × 药物分子图表示」计算每个诊断-药物对的注意力权重，
替代 case_study_plots.py 里的 np.random 模拟。
生成 saved/LLMDGTN/case_study/Fig8_heatmap.png（患者1 前两次就诊）。
"""
import os, sys, io, glob
import dill, numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LLMDGTN
from util import buildMPNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "mimic-iv")

# ---- 加载模型 + best 权重 ----
ehr_adj = np.array(dill.load(open(os.path.join(data_dir, "ehr_adj_final.pkl"), "rb")))
ddi_adj = np.array(dill.load(open(os.path.join(data_dir, "ddi_A_final.pkl"), "rb")))
ddi_mask_H = dill.load(open(os.path.join(data_dir, "ddi_mask_H.pkl"), "rb"))
molecule = dill.load(open(os.path.join(data_dir, "atc3toSMILES.pkl"), "rb"))
voc = dill.load(open(os.path.join(data_dir, "voc_final.pkl"), "rb"))
voc_size = (len(voc['diag_voc'].idx2word), len(voc['pro_voc'].idx2word), len(voc['med_voc'].idx2word))
med_map = {i: code for i, code in enumerate(molecule.keys())}
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_map, 2, device)

model = LLMDGTN(voc_size, ehr_adj, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint,
                average_projection, emb_dim=64, device=device,
                use_llm=True, use_transformer=True, use_mpnn=True)
model.to(device)
saved_dir = os.path.join(base, "saved", "LLMDGTN_opt")
h = dill.load(open(os.path.join(saved_dir, "history.pkl"), "rb"))
best_epoch = int(np.argmax(h['ja']))
ckpt = glob.glob(os.path.join(saved_dir, f"Epoch_{best_epoch}_*.model"))[0]
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print("加载权重:", os.path.basename(ckpt))

# ---- 加载 case_study_data（患者1 的诊断/药物）----
case_data = dill.load(open(os.path.join(base, "saved", "LLMDGTN", "case_study", "case_study_data.pkl"), 'rb'))
diag_voc = case_data['voc']['diag']
med_voc = case_data['voc']['med']
patient = case_data['patient_3visits']['visits']  # 患者1，3 visits

# ---- 药物表示：模型学到的分子图表示（MPNN_emb）----
drug_memory = model.MPNN_emb.detach()  # (n_drugs_with_mol, emb_dim)
drug_memory = F.normalize(drug_memory, p=2, dim=-1)  # L2 归一化
n_drug_repr = drug_memory.shape[0]
print(f"drug_memory shape: {tuple(drug_memory.shape)}")

# ---- 诊断嵌入层 ----
diag_emb_layer = model.embeddings[0]

def compute_attention(diag_indices, med_indices):
    """诊断嵌入 × 药物表示，逐诊断 softmax，得到 (n_diag, n_med) 注意力。"""
    diag_indices = [int(d) for d in diag_indices if int(d) < voc_size[0]]
    med_indices = [int(m) for m in med_indices if int(m) < n_drug_repr]
    if not diag_indices or not med_indices:
        return None, [], []
    diag_embs = diag_emb_layer(torch.LongTensor(diag_indices).to(device))  # (n_diag, dim)
    diag_embs = F.normalize(diag_embs, p=2, dim=-1)
    drug_sub = drug_memory[torch.LongTensor(med_indices).to(device)]  # (n_med, dim)
    logits = torch.mm(diag_embs, drug_sub.t())  # (n_diag, n_med)
    attn = torch.softmax(logits, dim=-1)  # 每行归一化到 0-1
    return attn.detach().cpu().numpy(), diag_indices, med_indices


# ---- 画热力图（患者1 前两次就诊）----
n_visits = 2
fig, axes = plt.subplots(1, n_visits, figsize=(7 * n_visits, 6))

for i in range(n_visits):
    visit = patient[i]
    all_meds = set()
    for v in patient[:i + 1]:
        all_meds.update(v['actual_meds_idx'])
    all_meds = sorted(all_meds)
    all_diags = visit['diagnoses_idx']

    attn, diag_list, med_list = compute_attention(all_diags, all_meds)
    if attn is None:
        continue
    ax = axes[i]
    im = ax.imshow(attn, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(diag_list)))
    ax.set_yticklabels([diag_voc.get(d, diag_voc.get(str(d), f'idx_{d}')) for d in diag_list], fontsize=8)
    ax.set_xticks(range(len(med_list)))
    ax.set_xticklabels([med_voc.get(m, med_voc.get(str(m), f'idx_{m}')) for m in med_list], rotation=90, fontsize=8)
    ax.set_title(f'({"a" if i == 0 else "b"}) Visit {i + 1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Medications', fontsize=10)
    ax.set_ylabel('Diagnoses', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = os.path.join(base, "saved", "LLMDGTN", "case_study", "Fig8_heatmap.png")
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("已保存真实热力图:", out)
