# -*- coding: utf-8 -*-
"""
图 6 真实注意力热力图（方案 A：梯度注意力 / Grad-CAM 风格）
用优化后 LLMDGTN 输出 logits 对诊断 token 嵌入的梯度，衡量「每个诊断对预测每个药物的贡献」，
得到真实的诊断-药物关联矩阵（替代 np.random 模拟）。
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
# 禁用 cudnn，让 GRU 用原生实现以支持 eval 模式反向（梯度注意力计算用）
torch.backends.cudnn.enabled = False
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
MPNNSet, Nf, avgp = buildMPNN(molecule, med_map, 2, device)

model = LLMDGTN(voc_size, ehr_adj, ddi_adj, ddi_mask_H, MPNNSet, Nf, avgp,
                emb_dim=64, device=device, use_llm=True, use_transformer=True, use_mpnn=True)
model.to(device)
sd = os.path.join(base, "saved", "LLMDGTN_opt")
h = dill.load(open(os.path.join(sd, "history.pkl"), "rb"))
be = int(np.argmax(h['ja']))
model.load_state_dict(torch.load(glob.glob(os.path.join(sd, f"Epoch_{be}_*.model"))[0], map_location=device))
model.eval()

# ---- 手动复现 forward（诊断 token 带 grad），与 models.py forward 完全一致 ----
def manual_forward(seq):
    i1_seq, i2_seq, i3_seq = [], [], []
    diag_token_embs = []
    for adm in seq:
        d_tensor = torch.LongTensor([max(0, min(int(x), voc_size[0] - 1)) for x in adm[0]]).to(device)
        d_emb = model.dropout(model.embeddings[0](d_tensor))
        d_emb.retain_grad()
        diag_token_embs.append(d_emb)
        i1_seq.append(d_emb.sum(0).unsqueeze(0).unsqueeze(0))

        p_tensor = torch.LongTensor([max(0, min(int(x), voc_size[1] - 1)) for x in adm[1]]).to(device)
        p_emb = model.dropout(model.embeddings[1](p_tensor))
        i2_seq.append(p_emb.sum(0).unsqueeze(0).unsqueeze(0))

        if len(adm[2]) > 0:
            m_tensor = torch.LongTensor([max(0, min(int(x), voc_size[2] - 1)) for x in adm[2]]).to(device)
            m_emb = model.dropout(model.embeddings[2](m_tensor))
            i3_seq.append(m_emb.sum(0).unsqueeze(0).unsqueeze(0))
        else:
            i3_seq.append(torch.zeros(1, 1, model.emb_dim).to(device))

    i1 = torch.cat(i1_seq, dim=1)
    i2 = torch.cat(i2_seq, dim=1)
    i3 = torch.cat(i3_seq, dim=1)
    o1, _ = model.encoders[0](i1)
    o2, _ = model.encoders[1](i2)
    o3, _ = model.m_encoders[0](i3)
    pr = torch.cat([o1, o2, o3], dim=-1).squeeze(0)
    queries = model.query(pr)
    query = queries[-1:]

    med_num = voc_size[2]
    drug_memory = model.MPNN_emb[:med_num]
    drug_memory = torch.nan_to_num(drug_memory)
    combined = drug_memory
    if model.use_llm:
        llm_proj = model.llm_proj(model.drug_llm_emb)
        llm_proj = torch.nan_to_num(llm_proj)
        llm_attn = torch.sigmoid(torch.mm(query, llm_proj.t()))
        llm_fact = torch.mm(llm_attn, llm_proj).expand(drug_memory.shape[0], -1)
        combined = drug_memory + model.llm_fuse_weight * llm_fact
    if model.use_transformer:
        drug_memory = model.transformer_encoder(combined.unsqueeze(0)).squeeze(0)
        drug_memory = torch.nan_to_num(drug_memory)
    else:
        drug_memory = combined

    key_weights = torch.sigmoid(torch.mm(query, drug_memory.t()))
    fact1 = torch.nan_to_num(torch.mm(key_weights, drug_memory))
    result = torch.nan_to_num(model.output(torch.cat([query, fact1], dim=-1)))
    return result, diag_token_embs


def grad_attention(seq, diag_indices, med_indices):
    """返回 (n_diag, n_med) 梯度注意力矩阵。"""
    result, diag_embs = manual_forward(seq)
    cur_diag = diag_embs[-1]  # 当前 visit 的诊断 token 嵌入
    n_diag = cur_diag.shape[0]
    attn = np.zeros((n_diag, len(med_indices)))
    for j, m in enumerate(med_indices):
        if m >= result.shape[1]:
            continue
        for de in diag_embs:
            de.grad = None
        result[0, m].backward(retain_graph=True)
        g = cur_diag.grad  # (n_diag, 64)，因诊断被 sum，各诊断梯度相同
        if g is not None:
            # Grad-CAM 风格：梯度 × 输入（诊断 token 嵌入），得到 per-token 贡献
            contrib = (g * cur_diag).sum(dim=-1)  # (n_diag,)
            attn[:, j] = contrib.detach().cpu().numpy()
    return attn


# ---- 读 case_study 数据，画患者1 前两次就诊热力图 ----
case = dill.load(open(os.path.join(base, "saved", "LLMDGTN", "case_study", "case_study_data.pkl"), 'rb'))
diag_voc = case['voc']['diag']
med_voc = case['voc']['med']
patient = case['patient_3visits']['visits']

# 原始患者序列（用于 forward）——从 records 里按诊断代码重定位
data = dill.load(open(os.path.join(data_dir, "records_final.pkl"), "rb"))
p1_diag_codes = ['3970','5854','4168','V0481','V4586','2449','3962','9971','53081']
p1_target = set(voc['diag_voc'].word2idx[c] for c in p1_diag_codes if c in voc['diag_voc'].word2idx)
p1_seq = None
for p in data:
    if len(p) >= 1 and p1_target.issubset(set(p[0][0])):
        p1_seq = p
        break
assert p1_seq is not None, "未找到患者1"

n_visits = 2
fig, axes = plt.subplots(1, n_visits, figsize=(7 * n_visits, 6))
for i in range(n_visits):
    visit = patient[i]
    all_meds = set()
    for v in patient[:i + 1]:
        all_meds.update(v['actual_meds_idx'])
    all_meds = sorted(all_meds)
    all_diags = visit['diagnoses_idx']

    seq = p1_seq[:i + 1]  # 前 i+1 次就诊
    attn = grad_attention(seq, all_diags, all_meds)
    if attn.size == 0:
        continue
    # 对称归一化到 [-1,1]：gradient×input 归因，红=增强预测的正贡献，蓝=抑制预测的负贡献
    max_abs = np.abs(attn).max()
    if max_abs > 0:
        attn = attn / max_abs

    ax = axes[i]
    im = ax.imshow(attn, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(len(all_diags)))
    ax.set_yticklabels([diag_voc.get(d, diag_voc.get(str(d), f'idx_{d}')) for d in all_diags], fontsize=8)
    ax.set_xticks(range(len(all_meds)))
    ax.set_xticklabels([med_voc.get(m, med_voc.get(str(m), f'idx_{m}')) for m in all_meds], rotation=90, fontsize=8)
    ax.set_title(f'({"a" if i == 0 else "b"}) Visit {i + 1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Medications', fontsize=10)
    ax.set_ylabel('Diagnoses', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = os.path.join(base, "saved", "LLMDGTN", "case_study", "Fig8_heatmap.png")
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("已保存真实梯度注意力热力图:", out)
