# -*- coding: utf-8 -*-
"""
用已训练好的最优 LLMDGTN（saved/LLMDGTN_opt/Epoch_47）在 eval 集上按就诊次数（1-10）
分组评估，更新 logs/results.csv 中 LLMDGTN 的行（其余模型 LR/GAMENet/SafeDrug 沿用旧值）。
"""
import os, sys, io, glob
import dill, numpy as np, torch
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LLMDGTN
from util import multi_label_metric, buildMPNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running device:", device)

# ---- 数据加载（同 LLMDGTN.py main）----
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "data", "mimic-iv")
ehr_adj = np.array(dill.load(open(os.path.join(data_dir, "ehr_adj_final.pkl"), "rb")))
ddi_adj = np.array(dill.load(open(os.path.join(data_dir, "ddi_A_final.pkl"), "rb")))
ddi_mask_H = dill.load(open(os.path.join(data_dir, "ddi_mask_H.pkl"), "rb"))
data = dill.load(open(os.path.join(data_dir, "records_final.pkl"), "rb"))
molecule = dill.load(open(os.path.join(data_dir, "atc3toSMILES.pkl"), "rb"))
voc = dill.load(open(os.path.join(data_dir, "voc_final.pkl"), "rb"))

split_point = int(len(data) * 2 / 3)
eval_len = int(len(data[split_point:]) / 2)
data_eval = data[split_point + eval_len:]

med_map = {i: code for i, code in enumerate(molecule.keys())}
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_map, 2, device)
voc_size = (len(voc['diag_voc'].idx2word), len(voc['pro_voc'].idx2word), len(voc['med_voc'].idx2word))
print("voc_size:", voc_size, "eval patients:", len(data_eval))

# ---- 构建模型 + 加载 best 权重 ----
model = LLMDGTN(voc_size, ehr_adj, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint,
                average_projection, emb_dim=64, device=device,
                use_llm=True, use_transformer=True, use_mpnn=True)
model.to(device)

saved_dir = os.path.join(base, "saved", "LLMDGTN_opt")
h = dill.load(open(os.path.join(saved_dir, "history.pkl"), "rb"))
best_epoch = int(np.argmax(h['ja']))
ckpt_files = glob.glob(os.path.join(saved_dir, f"Epoch_{best_epoch}_*.model"))
assert ckpt_files, f"未找到 Epoch_{best_epoch} 权重"
ckpt = ckpt_files[0]
model.load_state_dict(torch.load(ckpt, map_location=device))
print(f"加载权重: {os.path.basename(ckpt)}")
model.eval()


def calc_ddi_rate(pred_records, ddi_mat):
    ddi_cnt, pair_cnt = 0, 0
    for patient in pred_records:
        for meds in patient:
            meds = list(set(meds))
            for i in range(len(meds)):
                for j in range(i + 1, len(meds)):
                    pair_cnt += 1
                    if ddi_mat[meds[i], meds[j]] == 1:
                        ddi_cnt += 1
    return ddi_cnt / pair_cnt if pair_cnt > 0 else 0.0


def safe_seq(seq, mx):
    return [max(0, min(int(v), mx - 1)) for v in seq]


rows = []
with torch.no_grad():
    for target_visits in range(1, 11):
        ja_list, prauc_list, f1_list = [], [], []
        smm_record = []
        med_cnt, visit_cnt, patient_cnt = 0, 0, 0
        for input in data_eval:
            if len(input) != target_visits:
                continue
            patient_cnt += 1
            y_gt, y_pred, y_pred_prob = [], [], []
            for adm_idx, adm in enumerate(input):
                seq = []
                for s in input[:adm_idx + 1]:
                    seq.append([
                        safe_seq(s[0], voc_size[0]),
                        safe_seq(s[1], voc_size[1]),
                        safe_seq(s[2], voc_size[2]),
                        s[3] if len(s) > 3 else [],
                        s[4] if len(s) > 4 else [],
                    ])
                output, _ = model(seq)
                target_output = F.sigmoid(output).detach().cpu().numpy()[0]

                y_gt_tmp = np.zeros(voc_size[2])
                for item in adm[2]:
                    if 0 <= int(item) < voc_size[2]:
                        y_gt_tmp[int(item)] = 1
                y_gt.append(y_gt_tmp)
                y_pred_prob.append(target_output)

                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                y_label = sorted(np.where(y_pred_tmp == 1)[0])
                smm_record.append([y_label])
                visit_cnt += 1
                med_cnt += len(y_label)

            if len(y_gt) > 0:
                adm_ja, adm_prauc, _, _, adm_f1 = multi_label_metric(
                    np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                ja_list.append(adm_ja)
                prauc_list.append(adm_prauc)
                f1_list.append(adm_f1)

        ddi_rate = calc_ddi_rate(smm_record, ddi_adj)
        rows.append([
            'LLMDGTN', target_visits, patient_cnt,
            round(ddi_rate, 5),
            round(np.mean(ja_list) if ja_list else 0.0, 5),
            round(np.mean(prauc_list) if prauc_list else 0.0, 5),
            round(np.mean(f1_list) if f1_list else 0.0, 5),
            round(med_cnt / visit_cnt if visit_cnt > 0 else 0.0, 4),
        ])
        print(f"LLMDGTN visits={target_visits}: patients={patient_cnt} Jaccard={rows[-1][4]:.5f} "
              f"PRAUC={rows[-1][5]:.5f} DDI={rows[-1][3]:.5f}")

# ---- 合并写回 logs/results.csv（替换 LLMDGTN 旧行）----
csv_path = os.path.join(base, "..", "logs", "results.csv")
header = ['Model', 'Visits', 'Patients', 'DDI_Rate', 'Jaccard', 'PRAUC', 'AVG_F1', 'AVG_Med']
old_rows = []
if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        lines = f.read().strip().split('\n')
    if lines:
        for line in lines[1:]:
            parts = line.split(',')
            if parts and parts[0] != 'LLMDGTN':
                old_rows.append(line)

with open(csv_path, 'w', encoding='utf-8-sig') as f:
    f.write(','.join(header) + '\n')
    for line in old_rows:
        f.write(line + '\n')
    for r in rows:
        f.write(','.join(str(x) for x in r) + '\n')

print("\n已更新 logs/results.csv（LLMDGTN 行已替换，LR/GAMENet/SafeDrug 保留）")
