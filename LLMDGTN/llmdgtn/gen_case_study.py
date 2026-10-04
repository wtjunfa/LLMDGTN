# -*- coding: utf-8 -*-
"""
用优化后的 LLMDGTN（saved/LLMDGTN_opt 最优权重）生成案例研究数据：
  - 按论文 Table 6/7 的诊断代码精确定位患者1（3 visits）与患者2（1 visit）
  - 生成各就诊的 top-30 推荐药物及概率
  - 保存 case_study_data.pkl（供 case_study_plots.py 画图 7/8 + 表）
"""
import os, sys, io, glob
import dill, numpy as np, torch
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LLMDGTN
from util import buildMPNN, get_n_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running device:", device)

# ---- 数据加载 ----
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "mimic-iv")
ehr_adj = np.array(dill.load(open(os.path.join(data_dir, "ehr_adj_final.pkl"), "rb")))
ddi_adj = np.array(dill.load(open(os.path.join(data_dir, "ddi_A_final.pkl"), "rb")))
ddi_mask_H = dill.load(open(os.path.join(data_dir, "ddi_mask_H.pkl"), "rb"))
data = dill.load(open(os.path.join(data_dir, "records_final.pkl"), "rb"))
molecule = dill.load(open(os.path.join(data_dir, "atc3toSMILES.pkl"), "rb"))
voc = dill.load(open(os.path.join(data_dir, "voc_final.pkl"), "rb"))

diag_voc = voc['diag_voc']
pro_voc = voc['pro_voc']
med_voc = voc['med_voc']
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

med_map = {i: code for i, code in enumerate(molecule.keys())}
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_map, 2, device)

# ---- 论文里的患者诊断代码（Table 6/7）----
patient1_v1_diags = ['3970','5854','4168','V0481','V4586','2449','3962','9971','53081',
                     '42731','25000','72400','41401','45829','40390','E8782','V4365','2724','4019']
patient2_v1_diags = ['2762','27652','E8788','99592','2761','V103','72989','V4571','78552','03811','570','5849']

def to_idx(codes, w2i):
    out = []
    for c in codes:
        if c in w2i:
            out.append(w2i[c])
        elif str(c) in w2i:
            out.append(w2i[str(c)])
    return out

p1_target = set(to_idx(patient1_v1_diags, diag_voc.word2idx))
p2_target = set(to_idx(patient2_v1_diags, diag_voc.word2idx))

# ---- 定位患者 ----
patient1 = patient2 = None
for p in data:
    if len(p) >= 1 and p1_target and p1_target.issubset(set(p[0][0])):
        if patient1 is None:
            patient1 = p
    if len(p) >= 1 and p2_target and p2_target.issubset(set(p[0][0])):
        if patient2 is None:
            patient2 = p
    if patient1 is not None and patient2 is not None:
        break

assert patient1 is not None, "未找到患者1"
assert patient2 is not None, "未找到患者2"
print(f"患者1: {len(patient1)} visits | 患者2: {len(patient2)} visits")
assert len(patient1) == 3, "患者1 应为 3 visits"
assert len(patient2) == 1, "患者2 应为 1 visit"

# ---- 构建模型 + 加载 best 权重 ----
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
print(f"加载权重: {os.path.basename(ckpt)}")

# ---- 生成预测 ----
def safe_seq(seq, mx):
    return [max(0, min(int(v), mx - 1)) for v in seq]

def get_predictions(patient_visits):
    results = []
    with torch.no_grad():
        for adm_idx, adm in enumerate(patient_visits):
            seq = []
            for s in patient_visits[:adm_idx + 1]:
                seq.append([safe_seq(s[0], voc_size[0]), safe_seq(s[1], voc_size[1]),
                            safe_seq(s[2], voc_size[2]),
                            s[3] if len(s) > 3 else [], s[4] if len(s) > 4 else []])
            out = model(seq)
            target_output = out[0] if isinstance(out, tuple) else out
            probs = F.sigmoid(target_output).detach().cpu().numpy()[0]
            actual_meds = sorted([int(m) for m in adm[2]])
            top30_idx = np.argsort(probs)[::-1][:30]
            results.append({
                'visit_idx': adm_idx,
                'diagnoses': [int(d) for d in adm[0]],
                'procedures': [int(p) for p in adm[1]],
                'actual_meds': actual_meds,
                'top_meds_idx': top30_idx.tolist(),
                'top_meds_probs': probs[top30_idx].tolist(),
                'all_probs': probs.tolist(),
            })
    return results

print("\n[生成预测]")
r3 = get_predictions(patient1)
r1 = get_predictions(patient2)

# ---- 保存 ----
OUTPUT_DIR = os.path.join(base, "saved", "LLMDGTN", "case_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_visit_dict(r):
    return {
        'diagnoses_idx': r['diagnoses'],
        'procedures_idx': r['procedures'],
        'actual_meds_idx': r['actual_meds'],
        'top_meds_idx': r['top_meds_idx'],
        'top_meds_probs': r['top_meds_probs'],
        'all_probs': r['all_probs'],
    }

output_data = {
    'patient_3visits': {'index': 0, 'visits': [to_visit_dict(r) for r in r3]},
    'patient_1visit': {'index': 0, 'visits': [to_visit_dict(r) for r in r1]},
    'voc': {
        'diag': diag_voc.idx2word,
        'proc': pro_voc.idx2word,
        'med': med_voc.idx2word,
    },
    'config': {'emb_dim': 64, 'epochs': best_epoch + 1, 'seed': 727},
}
with open(os.path.join(OUTPUT_DIR, 'case_study_data.pkl'), 'wb') as f:
    dill.dump(output_data, f)
print(f"\n已保存: {OUTPUT_DIR}/case_study_data.pkl")

# ---- 打印摘要 ----
for name, rs in [("患者1 (3 visits)", r3), ("患者2 (1 visit)", r1)]:
    print(f"\n{name}:")
    for r in rs:
        actual = set(r['actual_meds'])
        top5 = r['top_meds_idx'][:5]
        hit = [m for m in top5 if m in actual]
        print(f"  Visit {r['visit_idx']+1}: 实际{len(actual)}种药, top5命中{len(hit)}个, "
              f"top5概率{[f'{p:.3f}' for p in r['top_meds_probs'][:5]]}")
