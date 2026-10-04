# -*- coding: utf-8 -*-
"""
用最终 best 模型对 LR / GAMENet / SafeDrug 做按就诊次数（1-10）评估，
更新 logs/results.csv 中这三个模型的行（LLMDGTN 行保留已更新的结果）。
LR 重新训练（sklearn）；GAMENet/SafeDrug 加载 saved/*/best.model。
"""
import os, sys, io
import dill, numpy as np, torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GAMENet, SafeDrugModel
from util import multi_label_metric, buildMPNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running device:", device)

# ---- 数据加载（不 shuffle，split 与 LLMDGTN 一致）----
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "..", "..", "data", "mimic-iv")
ehr_adj = np.array(dill.load(open(os.path.join(data_dir, "ehr_adj_final.pkl"), "rb")))
ddi_adj = np.array(dill.load(open(os.path.join(data_dir, "ddi_A_final.pkl"), "rb")))
ddi_mask_H = dill.load(open(os.path.join(data_dir, "ddi_mask_H.pkl"), "rb"))
data = dill.load(open(os.path.join(data_dir, "records_final.pkl"), "rb"))
molecule = dill.load(open(os.path.join(data_dir, "atc3toSMILES.pkl"), "rb"))
voc = dill.load(open(os.path.join(data_dir, "voc_final.pkl"), "rb"))

diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

split_point = int(len(data) * 2 / 3)
eval_len = int(len(data[split_point:]) / 2)
data_train = data[:split_point]
data_eval = data[split_point + eval_len:]
print("voc_size:", voc_size, "eval patients:", len(data_eval))


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


def agg_metrics(y_gt_list, y_pred_list, y_prob_list):
    """对一批样本计算平均指标"""
    if not y_gt_list:
        return 0.0, 0.0, 0.0
    ja, prauc, _, _, f1 = multi_label_metric(
        np.array(y_gt_list), np.array(y_pred_list), np.array(y_prob_list))
    return ja, prauc, f1


def visit_loop(model, predict_fn, name):
    """按就诊次数遍历评估。predict_fn(adm_seq, visit) -> prob 向量 (med_num,)"""
    rows = []
    with torch.no_grad():
        for target_visits in range(1, 11):
            ja_list, prauc_list, f1_list = [], [], []
            smm_record = []
            med_cnt, visit_cnt, patient_cnt = 0, 0, 0
            for patient in data_eval:
                if len(patient) != target_visits:
                    continue
                patient_cnt += 1
                y_gt, y_pred, y_prob = [], [], []
                for adm_idx, visit in enumerate(patient):
                    prob = predict_fn(patient[:adm_idx + 1], visit)
                    y_gt_tmp = np.zeros(voc_size[2])
                    for item in visit[2]:
                        if 0 <= int(item) < voc_size[2]:
                            y_gt_tmp[int(item)] = 1
                    y_gt.append(y_gt_tmp)
                    y_prob.append(prob)
                    y_pred_tmp = (prob >= 0.5).astype(int)
                    y_pred.append(y_pred_tmp)
                    y_label = np.where(y_pred_tmp == 1)[0]
                    smm_record.append([y_label.tolist()])
                    visit_cnt += 1
                    med_cnt += len(y_label)
                if y_gt:
                    ja, prauc, f1 = agg_metrics(y_gt, y_pred, y_prob)
                    ja_list.append(ja); prauc_list.append(prauc); f1_list.append(f1)
            ddi_rate = calc_ddi_rate(smm_record, ddi_adj)
            rows.append([name, target_visits, patient_cnt,
                         round(ddi_rate, 5), round(float(np.mean(ja_list)), 5),
                         round(float(np.mean(prauc_list)), 5), round(float(np.mean(f1_list)), 5),
                         round(med_cnt / visit_cnt if visit_cnt else 0.0, 4)])
            print(f"{name} visits={target_visits}: patients={patient_cnt} Jaccard={rows[-1][4]:.5f} PRAUC={rows[-1][5]:.5f}")
    return rows


# ================= LR =================
print("\n===== LR（重新训练）=====")
def lr_feature(visit):
    x = np.zeros(voc_size[0] + voc_size[1])
    for d in visit[0]:
        if 0 <= int(d) < voc_size[0]:
            x[int(d)] = 1
    for p in visit[1]:
        if 0 <= int(p) < voc_size[1]:
            x[voc_size[0] + int(p)] = 1
    return x

X_train, y_train = [], []
for patient in data_train:
    for visit in patient:
        x = lr_feature(visit)
        y = np.zeros(voc_size[2])
        for m in visit[2]:
            if 0 <= int(m) < voc_size[2]:
                y[int(m)] = 1
        X_train.append(x); y_train.append(y)
lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr.fit(np.array(X_train), np.array(y_train))
print("LR 训练完成")

def lr_predict(adm_seq, visit):
    return lr.predict_proba([lr_feature(visit)])[0]

lr_rows = visit_loop(lr, lr_predict, 'LR')

# ================= GAMENet =================
print("\n===== GAMENet（加载 best.model）=====")
gamenet = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=False)
gamenet.load_state_dict(torch.load(os.path.join(base, "saved", "GAMENet", "best.model"), map_location=device))
gamenet.to(device)
gamenet.eval()

def gamenet_predict(adm_seq, visit):
    seq = []
    for s in adm_seq:
        seq.append([safe_seq(s[0], voc_size[0]), safe_seq(s[1], voc_size[1]),
                    safe_seq(s[2], voc_size[2])])
    out = gamenet(seq)
    return F.sigmoid(out).detach().cpu().numpy()[0]

gamenet_rows = visit_loop(gamenet, gamenet_predict, 'GAMENet')

# ================= SafeDrug =================
print("\n===== SafeDrug（加载 best.model）=====")
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)
safedrug = SafeDrugModel(voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint,
                         average_projection, emb_dim=64, device=device)
safedrug.load_state_dict(torch.load(os.path.join(base, "saved", "SafeDrug", "best.model"), map_location=device))
safedrug.to(device)
safedrug.eval()

def safedrug_predict(adm_seq, visit):
    seq = []
    for s in adm_seq:
        seq.append([safe_seq(s[0], voc_size[0]), safe_seq(s[1], voc_size[1]),
                    safe_seq(s[2], voc_size[2]), s[3] if len(s) > 3 else [],
                    s[4] if len(s) > 4 else []])
    out = safedrug(seq)
    result = out[0] if isinstance(out, tuple) else out
    return F.sigmoid(result).detach().cpu().numpy()[0]

safedrug_rows = visit_loop(safedrug, safedrug_predict, 'SafeDrug')

# ================= 合并写回 results.csv =================
csv_path = os.path.join(base, "..", "..", "logs", "results.csv")
header = ['Model', 'Visits', 'Patients', 'DDI_Rate', 'Jaccard', 'PRAUC', 'AVG_F1', 'AVG_Med']
new_map = {}
for r in lr_rows + gamenet_rows + safedrug_rows:
    new_map[(r[0], r[1])] = r

# 保留 LLMDGTN 行 + 其他未知行，替换 LR/GAMENet/SafeDrug
lines = []
if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        raw = f.read().strip().split('\n')
    if raw and raw[0].startswith('Model'):
        for line in raw[1:]:
            parts = line.split(',')
            if parts and parts[0] in ('LR', 'GAMENet', 'SafeDrug'):
                continue  # 将被替换
            lines.append(line)

with open(csv_path, 'w', encoding='utf-8-sig') as f:
    f.write(','.join(header) + '\n')
    for line in lines:
        f.write(line + '\n')
    for r in lr_rows + gamenet_rows + safedrug_rows:
        f.write(','.join(str(x) for x in r) + '\n')

print("\n已更新 logs/results.csv（LR/GAMENet/SafeDrug 已替换，LLMDGTN 保留）")
