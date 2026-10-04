"""
Case Study Experiment for LLMDGTN
=================================
Reproduces Tables 7-9 and Figs 8-10 from the paper.

Pipeline:
1. Quick training (5 epochs) to get a model
2. Select 2 patients: one with 3 visits, one with 1 visit
3. Generate:
   - Table 7: First patient (3 visits) - diagnoses/procedures/medications
   - Table 8: Second patient (1 visit) - diagnoses/procedures/medications
   - Fig 8: Dynamic network attention heatmap (2 subplots)
   - Fig 9: Recommended medications for first patient (visit 3)
   - Fig 10: Recommended medications for second patient
   - Table 9: Medication classification (Refilled vs New)
"""

import os
import sys
import json
import pickle
import dill
import numpy as np
import torch
import torch.nn.functional as F
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LLMDGTN
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running device: {device}")

# ============================================================
# 配置
# ============================================================
EMB_DIM = 64
LR = 5e-4
EPOCH = 5
TARGET_DDI = 0.06
KP = 0.05
SEED = 1208

OUTPUT_DIR = os.path.join("saved", "LLMDGTN", "case_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 数据加载 (与 exp_ddi_sensitivity.py 一致)
# ============================================================

def calc_ddi_rate(pred_records, ddi_mat):
    import numpy as np
    ddi_cnt = 0
    pair_cnt = 0
    for patient in pred_records:
        for meds in patient:
            meds = list(set(meds))
            for i in range(len(meds)):
                for j in range(i + 1, len(meds)):
                    pair_cnt += 1
                    if ddi_mat[meds[i], meds[j]] == 1:
                        ddi_cnt += 1
    return ddi_cnt / pair_cnt if pair_cnt > 0 else 0.0


def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output")
    data_path = os.path.join(data_dir, "records_final.pkl")
    voc_path = os.path.join(data_dir, "voc_final.pkl")
    ehr_adj_path = os.path.join(data_dir, "ehr_adj_final.pkl")
    ddi_adj_path = os.path.join(data_dir, "ddi_A_final.pkl")
    ddi_mask_path = os.path.join(data_dir, "ddi_mask_H.pkl")
    molecule_path = os.path.join(data_dir, "atc3toSMILES.pkl")

    print("Loading data...")
    ehr_adj = np.array(dill.load(open(ehr_adj_path, "rb")))
    ddi_adj = np.array(dill.load(open(ddi_adj_path, "rb")))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    def load_vocab(path):
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    voc = load_vocab(voc_path)
    diag_voc = voc["diag_voc"]
    pro_voc = voc["pro_voc"]
    med_voc = voc["med_voc"]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    med_map = {i: code for i, code in enumerate(molecule.keys())}
    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_map, 2, device)

    voc_size = (
        len(voc['diag_voc']['idx2word']),
        len(voc['pro_voc']['idx2word']),
        len(voc['med_voc']['idx2word'])
    )
    print(f"Vocabulary size: diag={voc_size[0]}, pro={voc_size[1]}, med={voc_size[2]}")
    print(f"Train: {len(data_train)}, Eval: {len(data_eval)}, Test: {len(data_test)}")

    return {
        'ehr_adj': ehr_adj, 'ddi_adj': ddi_adj, 'ddi_mask_H': ddi_mask_H,
        'data_train': data_train, 'data_eval': data_eval, 'data_test': data_test,
        'MPNNSet': MPNNSet, 'N_fingerprint': N_fingerprint,
        'average_projection': average_projection, 'voc_size': voc_size,
        'voc': voc,
    }


# ============================================================
# 快速训练模型
# ============================================================

def quick_train(model, data_train, data_eval, voc_size, ddi_adj, epochs=5):
    """Quick training of LLMDGTN for case study."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n[Training] {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        np.random.seed(SEED + epoch)
        np.random.shuffle(data_train)
        tic = time.time()

        for step, input in enumerate(data_train):
            loss = 0
            for idx, adm in enumerate(input):
                seq_input = input[: idx + 1]

                def safe_seq(seq, max_size):
                    return [max(0, min(int(v), max_size - 1)) for v in seq]

                for i, step_adm in enumerate(seq_input):
                    seq_input[i] = [
                        safe_seq(step_adm[0], voc_size[0]),
                        safe_seq(step_adm[1], voc_size[1]),
                        safe_seq(step_adm[2], voc_size[2]),
                        step_adm[3] if len(step_adm) > 3 else [],
                        step_adm[4] if len(step_adm) > 4 else []
                    ]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for item in adm[2]:
                    safe_item = max(0, min(int(item), voc_size[2] - 1))
                    loss_multi_target[0][safe_item] = 1

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result_sigmoid = F.sigmoid(result).detach().cpu().numpy()[0]
                result_sigmoid[result_sigmoid >= 0.5] = 1
                result_sigmoid[result_sigmoid < 0.5] = 0
                y_label = np.where(result_sigmoid == 1)[0]
                current_ddi_rate = calc_ddi_rate([[y_label]], ddi_adj)

                if current_ddi_rate <= TARGET_DDI:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0.0, 1 + (TARGET_DDI - current_ddi_rate) / KP)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            if step % 500 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(data_train)}, "
                      f"loss={loss.item():.4f}, time={time.time()-tic:.1f}s", flush=True)

        print(f"  epoch {epoch+1} done, time={time.time()-tic:.1f}s")

    print("[Training] Complete!")
    return model


# ============================================================
# 生成推荐结果
# ============================================================

def get_predictions(model, patient_visits, voc_size):
    """Get predictions for each visit of a patient."""
    model.eval()
    results = []

    with torch.no_grad():
        for adm_idx, adm in enumerate(patient_visits):
            seq_input = patient_visits[: adm_idx + 1]

            def safe_seq(seq, max_size):
                return [max(0, min(int(v), max_size - 1)) for v in seq]

            for i, step_adm in enumerate(seq_input):
                seq_input[i] = [
                    safe_seq(step_adm[0], voc_size[0]),
                    safe_seq(step_adm[1], voc_size[1]),
                    safe_seq(step_adm[2], voc_size[2]),
                    step_adm[3] if len(step_adm) > 3 else [],
                    step_adm[4] if len(step_adm) > 4 else []
                ]

            target_output, _ = model(seq_input)
            probs = F.sigmoid(target_output).detach().cpu().numpy()[0]

            # Ground truth
            actual_meds = sorted([m for m in adm[2]])

            # Get top 30 medications
            top30_idx = np.argsort(probs)[::-1][:30]
            top30_probs = probs[top30_idx]
            top30_meds = top30_idx.tolist()

            results.append({
                'visit_idx': adm_idx,
                'diagnoses': adm[0],
                'procedures': adm[1],
                'actual_meds': actual_meds,
                'top_meds_idx': top30_meds,
                'top_meds_probs': top30_probs.tolist(),
                'all_probs': probs.tolist(),
            })

    return results


# ============================================================
# 查找测试集中的患者
# ============================================================

def find_patients(data_test):
    """Find one patient with 3 visits and one with 1 visit."""
    patient_3visits = None
    patient_1visit = None

    for idx, patient in enumerate(data_test):
        if len(patient) == 3 and patient_3visits is None:
            if all(len(adm[2]) > 0 for adm in patient):
                patient_3visits = (idx, patient)
        elif len(patient) == 1 and patient_1visit is None:
            if len(patient[0][2]) > 0:
                patient_1visit = (idx, patient)

        if patient_3visits is not None and patient_1visit is not None:
            break

    return patient_3visits, patient_1visit


# ============================================================
# 主函数
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. 加载数据
    bundle = load_data()
    voc_size = bundle['voc_size']
    voc = bundle['voc']
    ddi_adj = bundle['ddi_adj']
    data_train = bundle['data_train']
    data_test = bundle['data_test']

    diag_voc = voc['diag_voc']['idx2word']
    proc_voc = voc['pro_voc']['idx2word']
    med_voc = voc['med_voc']['idx2word']

    # 2. 创建模型
    model = LLMDGTN(
        voc_size,
        bundle['ehr_adj'],
        ddi_adj,
        bundle['ddi_mask_H'],
        bundle['MPNNSet'],
        bundle['N_fingerprint'],
        bundle['average_projection'],
        emb_dim=EMB_DIM,
        device=device,
    )
    model.to(device)
    print(f"Model parameters: {get_n_params(model):,}")

    # 3. 快速训练
    model = quick_train(model, data_train, bundle['data_eval'], voc_size, ddi_adj, epochs=EPOCH)

    # 4. 寻找合适患者
    print("\n[Patient Selection]")
    patient_3visits, patient_1visit = find_patients(data_test)
    if patient_3visits:
        print(f"3-visit patient index: {patient_3visits[0]}")
    if patient_1visit:
        print(f"1-visit patient index: {patient_1visit[0]}")

    if not patient_3visits or not patient_1visit:
        print("ERROR: Could not find suitable patients!")
        return

    # 5. 生成推荐结果
    print("\n[Generating Predictions]")
    results_3 = get_predictions(model, patient_3visits[1], voc_size)
    results_1 = get_predictions(model, patient_1visit[1], voc_size)

    # 6. 保存结果
    output_data = {
        'patient_3visits': {
            'index': patient_3visits[0],
            'visits': [{
                'diagnoses_idx': r['diagnoses'],
                'procedures_idx': r['procedures'],
                'actual_meds_idx': r['actual_meds'],
                'top_meds_idx': r['top_meds_idx'],
                'top_meds_probs': r['top_meds_probs'],
                'all_probs': r['all_probs'],
            } for r in results_3]
        },
        'patient_1visit': {
            'index': patient_1visit[0],
            'visits': [{
                'diagnoses_idx': r['diagnoses'],
                'procedures_idx': r['procedures'],
                'actual_meds_idx': r['actual_meds'],
                'top_meds_idx': r['top_meds_idx'],
                'top_meds_probs': r['top_meds_probs'],
                'all_probs': r['all_probs'],
            } for r in results_1]
        },
        'voc': {
            'diag': diag_voc,
            'proc': proc_voc,
            'med': med_voc,
        },
        'config': {
            'emb_dim': EMB_DIM,
            'lr': LR,
            'epochs': EPOCH,
            'target_ddi': TARGET_DDI,
            'seed': SEED,
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'case_study_data.pkl'), 'wb') as f:
        dill.dump(output_data, f)
    print(f"\nData saved: {os.path.join(OUTPUT_DIR, 'case_study_data.pkl')}")

    # 7. 输出统计信息
    print("\n" + "="*80)
    print("CASE STUDY DATA SUMMARY")
    print("="*80)

    for name, results in [("Patient 1 (3 visits)", results_3), ("Patient 2 (1 visit)", results_1)]:
        print(f"\n{name}:")
        for r in results:
            print(f"  Visit {r['visit_idx']+1}:")
            print(f"    Diagnoses ({len(r['diagnoses'])}): {r['diagnoses'][:5]}...")
            print(f"    Procedures ({len(r['procedures'])}): {r['procedures'][:5]}...")
            print(f"    Actual meds ({len(r['actual_meds'])}): {r['actual_meds'][:5]}...")
            print(f"    Top-5 predicted idx: {r['top_meds_idx'][:5]}")
            print(f"    Top-5 predicted prob: {[f'{p:.4f}' for p in r['top_meds_probs'][:5]]}")

    print("\n[Done] Run case_study_plots.py to generate tables and figures.")


if __name__ == '__main__':
    main()
