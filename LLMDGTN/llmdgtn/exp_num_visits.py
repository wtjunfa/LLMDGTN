r"""
Number of Visits Sensitivity Experiment
==========================================
Evaluate how model performance varies with different numbers of patient visits.

This experiment:
1. Groups test patients by their number of visits (e.g., 1-5, 6-10, 11-15, ...)
2. Evaluates the trained LLMDGTN model on each group separately
3. Records metrics (Jaccard, F1, PRAUC, DDI, etc.) per visit group
4. Generates results table and plot for paper

Usage:
    cd to this directory
    python exp_num_visits.py

Output:
    saved/LLMDGTN/num_visits_experiment/
        - results.csv          full results per visit group
        - summary_table.txt    paper-style summary table
        - Fig3.png             performance curve plot
"""

import os
import sys
import json
import pickle
import dill
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LLMDGTN
from util import (
    llprint, multi_label_metric, ddi_rate_score,
    get_n_params, buildMPNN
)

# ============================================================
# 固定随机种子和超参数
# ============================================================
SEEDS = [1208, 2053]
EPOCH = 50
TARGET_DDI = 0.06
KP = 0.05
LR = 5e-4
EMB_DIM = 64  # 使用最优维度

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running device: {device}")

# 就诊次数分组设置 (左闭右开区间)
VISIT_BINS = [1, 3, 5, 7, 10, 15, 30, 100]  # 分组边界
VISIT_LABELS = ["1-2", "3-4", "5-6", "7-9", "10-14", "15-29", "≥30"]


# ============================================================
# 数据加载辅助函数
# ============================================================

def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    med_freqs = {}
    dm_freqs = {}
    pm_freqs = {}
    total_visit = 0
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()
        if train_key_set is not None and seqex not in train_key_set:
            total_visit += len(seqex)
            continue
        for key in seqex:
            dx_ids = key[0]
            proc_ids = key[1]
            med_ids = key[2]
            for dx in dx_ids:
                if dx not in dx_freqs:
                    dx_freqs[dx] = 0
                dx_freqs[dx] += 1
            for proc in proc_ids:
                if proc not in proc_freqs:
                    proc_freqs[proc] = 0
                proc_freqs[proc] += 1
            for med in med_ids:
                if med not in med_freqs:
                    med_freqs[med] = 0
                med_freqs[med] += 1
            for dx in dx_ids:
                for med in med_ids:
                    dm = str(dx) + ',' + str(med)
                    if dm not in dm_freqs:
                        dm_freqs[dm] = 0
                    dm_freqs[dm] += 1
            for proc in proc_ids:
                for med in med_ids:
                    pm = str(proc) + ',' + str(med)
                    if pm not in pm_freqs:
                        pm_freqs[pm] = 0
                    pm_freqs[pm] += 1
            total_visit += 1

    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()])
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()])
    med_probs = dict([(k, v / float(total_visit)) for k, v in med_freqs.items()])
    dm_probs = dict([(k, v / float(total_visit)) for k, v in dm_freqs.items()])
    pm_probs = dict([(k, v / float(total_visit)) for k, v in pm_freqs.items()])

    dm_cond_probs = {}
    md_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for med, med_prob in med_probs.items():
            dm = str(dx) + ',' + str(med)
            md = str(med) + ',' + str(dx)
            if dm in dm_probs:
                dm_cond_probs[dm] = dm_probs[dm] / dx_prob
                md_cond_probs[md] = dm_probs[dm] / med_prob
            else:
                dm_cond_probs[dm] = 0.0
                md_cond_probs[md] = 0.0

    pm_cond_probs = {}
    mp_cond_probs = {}
    for proc, proc_prob in proc_probs.items():
        for med, med_prob in med_probs.items():
            pm = str(proc) + ',' + str(med)
            mp = str(med) + ',' + str(proc)
            if pm in pm_probs:
                pm_cond_probs[pm] = pm_probs[pm] / proc_prob
                mp_cond_probs[mp] = pm_probs[pm] / med_prob
            else:
                pm_cond_probs[pm] = 0.0
                mp_cond_probs[mp] = 0.0

    pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
    pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'), -1)
    pickle.dump(med_probs, open(output_path + '/med_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_probs, open(output_path + '/dm_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_cond_probs, open(output_path + '/dm_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(md_cond_probs, open(output_path + '/md_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(pm_cond_probs, open(output_path + '/pm_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(mp_cond_probs, open(output_path + '/mp_probs.empirical.p', 'wb'), -1)


def add_sparse_prior_guide_dp(seqex_list, stats_path, key_set=None):
    print('Loading conditional probabilities.')
    dm_cond_probs = pickle.load(open(stats_path + '/dm_cond_probs.empirical.p', 'rb'))
    md_cond_probs = pickle.load(open(stats_path + '/md_cond_probs.empirical.p', 'rb'))
    pm_cond_probs = pickle.load(open(stats_path + '/pm_cond_probs.empirical.p', 'rb'))
    mp_cond_probs = pickle.load(open(stats_path + '/mp_cond_probs.empirical.p', 'rb'))

    print('Adding prior guide.')
    total_visit = 0
    new_seqex_list = []

    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()
        if key_set is not None and seqex not in key_set:
            total_visit += len(seqex)
            continue
        for key in seqex:
            dx_ids = key[0]
            proc_idx = key[1]
            med_ids = key[2]

            indices_dpm = []
            values_dpm = []

            for i, dx in enumerate(dx_ids):
                for j, med in enumerate(med_ids):
                    dm = str(dx) + ',' + str(med)
                    indices_dpm.append((i, len(dx_ids) + len(proc_idx) + j))
                    prob = 0.0 if dm not in dm_cond_probs else dm_cond_probs[dm]
                    values_dpm.append(prob)

            for i, proc in enumerate(proc_idx):
                for j, med in enumerate(med_ids):
                    pm = str(proc) + ',' + str(med)
                    indices_dpm.append((len(dx_ids) + i, len(dx_ids) + len(proc_idx) + j))
                    prob = 0.0 if pm not in pm_cond_probs else pm_cond_probs[pm]
                    values_dpm.append(prob)

            for i, med in enumerate(med_ids):
                for j, dx in enumerate(dx_ids):
                    md = str(med) + ',' + str(dx)
                    indices_dpm.append((len(dx_ids) + len(proc_idx) + med, j))
                    prob = 0.0 if md not in md_cond_probs else md_cond_probs[md]
                    values_dpm.append(prob)

                for j, proc in enumerate(proc_idx):
                    mp = str(med) + ',' + str(proc)
                    indices_dpm.append((len(dx_ids) + len(proc_idx) + med, len(dx_ids) + j))
                    prob = 0.0 if mp not in mp_cond_probs else mp_cond_probs[mp]
                    values_dpm.append(prob)

            key.append(indices_dpm)
            key.append(values_dpm)
            total_visit += 1


# ============================================================
# 数据加载
# ============================================================

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

    # 统计每个患者的就诊次数分布
    visit_counts = [len(patient) for patient in data_eval]
    print(f"\nEval set visit count statistics:")
    print(f"  Min visits: {min(visit_counts)}")
    print(f"  Max visits: {max(visit_counts)}")
    print(f"  Mean visits: {np.mean(visit_counts):.1f}")
    print(f"  Median visits: {np.median(visit_counts):.0f}")

    # 打印各分组的样本数量
    for i in range(len(VISIT_BINS) - 1):
        low, high = VISIT_BINS[i], VISIT_BINS[i + 1]
        count = sum(1 for v in visit_counts if low <= v < high)
        label = VISIT_LABELS[i]
        print(f"  Visits [{label}): {count} patients ({count/len(visit_counts)*100:.1f}%)")

    return {
        'ehr_adj': ehr_adj, 'ddi_adj': ddi_adj, 'ddi_mask_H': ddi_mask_H,
        'data_train': data_train, 'data_eval': data_eval, 'data_test': data_test,
        'MPNNSet': MPNNSet, 'N_fingerprint': N_fingerprint,
        'average_projection': average_projection, 'voc_size': voc_size,
        'ddi_adj': ddi_adj,
    }


def calc_ddi_rate(pred_records, ddi_mat):
    """计算推荐结果中的 DDI 率"""
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


# ============================================================
# 按就诊次数分组评估函数
# ============================================================

def eval_model_by_visits(model, data_eval, voc_size, ddi_adj, visit_bins=VISIT_BINS):
    """
    评估模型性能，按就诊次数分组返回结果
    
    Returns:
        dict: 每个分组的评估结果
              {
                  '1-2': {'ddi_rate': ..., 'jaccard': ..., ...},
                  '3-4': {...},
                  ...
              }
    """
    model.eval()
    
    # 初始化每个分组的结果容器
    group_results = {}
    for i in range(len(visit_bins) - 1):
        label = VISIT_LABELS[i]
        group_results[label] = {
            'smm_record': [],
            'ja_all': [], 'prauc_all': [], 'avg_p_all': [],
            'avg_r_all': [], 'avg_f1_all': [],
            'med_cnt': 0, 'visit_cnt': 0,
            'patient_cnt': 0,
        }
    
    with torch.no_grad():
        for step, input in enumerate(data_eval):
            n_visits = len(input)  # 该患者的就诊次数
            
            # 确定属于哪个分组
            group_label = None
            for i in range(len(visit_bins) - 1):
                if visit_bins[i] <= n_visits < visit_bins[i + 1]:
                    group_label = VISIT_LABELS[i]
                    break
            
            if group_label is None:
                continue  # 超出范围的跳过
            
            grp = group_results[group_label]
            grp['patient_cnt'] += 1
            
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            
            for adm_idx, adm in enumerate(input):
                target_output, _ = model(input[: adm_idx + 1])
                
                y_gt_tmp = np.zeros(voc_size[2])
                for item in adm[2]:
                    y_gt_tmp[item] = 1
                y_gt.append(y_gt_tmp)
                
                target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
                y_pred_prob.append(target_output)
                
                y_pred_tmp = target_output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)
                
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                grp['visit_cnt'] += 1
                grp['med_cnt'] += len(y_pred_label_tmp)
            
            grp['smm_record'].append(y_pred_label)
            
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
            )
            grp['ja_all'].append(adm_ja)
            grp['prauc_all'].append(adm_prauc)
            grp['avg_p_all'].append(adm_avg_p)
            grp['avg_r_all'].append(adm_avg_r)
            grp['avg_f1_all'].append(adm_avg_f1)
            
            llprint("\reval step: {} / {}".format(step, len(data_eval)))
    
    # 汇总每个分组的指标
    summary = {}
    for label in group_results:
        grp = group_results[label]
        
        if grp['visit_cnt'] == 0:
            summary[label] = {
                'ddi_rate': 0, 'jaccard': 0, 'prauc': 0,
                'avg_p': 0, 'avg_r': 0, 'avg_f1': 0, 'avg_med': 0,
                'patient_cnt': 0, 'visit_cnt': 0,
            }
            continue
        
        ddi_rate = calc_ddi_rate(grp['smm_record'], ddi_adj)
        
        summary[label] = {
            'ddi_rate': ddi_rate,
            'jaccard': np.mean(grp['ja_all']),
            'prauc': np.mean(grp['prauc_all']),
            'avg_p': np.mean(grp['avg_p_all']),
            'avg_r': np.mean(grp['avg_r_all']),
            'avg_f1': np.mean(grp['avg_f1_all']),
            'avg_med': grp['med_cnt'] / grp['visit_cnt'],
            'patient_cnt': grp['patient_cnt'],
            'visit_cnt': grp['visit_cnt'],
        }
    
    return summary


# ============================================================
# 训练函数（从 exp_emb_dim.py 复制）
# ============================================================

def train_and_evaluate(seed, data_bundle, epochs=EPOCH):
    """训练一轮模型，返回最佳模型和指标"""
    torch.cuda.empty_cache()
    gc.collect()
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    ehr_adj = data_bundle['ehr_adj']
    ddi_adj = data_bundle['ddi_adj']
    ddi_mask_H = data_bundle['ddi_mask_H']
    data_train = data_bundle['data_train']
    data_eval = data_bundle['data_eval']
    MPNNSet = data_bundle['MPNNSet']
    N_fingerprint = data_bundle['N_fingerprint']
    average_projection = data_bundle['average_projection']
    voc_size = data_bundle['voc_size']

    model = LLMDGTN(
        voc_size, ehr_adj, ddi_adj, ddi_mask_H,
        MPNNSet, N_fingerprint, average_projection,
        emb_dim=EMB_DIM, device=device,
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    
    best_ja = 0
    best_metrics = None
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"  Training with seed={seed}, dim={EMB_DIM}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        tic = time.time()
        model.train()

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
                        step_adm[4] if len(step_adm) > 4 else [],
                    ]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for item in adm[2]:
                    safe_item = max(0, min(int(item), voc_size[2] - 1))
                    loss_multi_target[0][safe_item] = 1

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result_np = F.sigmoid(result).detach().cpu().numpy()[0]
                result_np[result_np >= 0.5] = 1
                result_np[result_np < 0.5] = 0
                y_label = np.where(result_np == 1)[0]
                current_ddi_rate = calc_ddi_rate([[y_label]], ddi_adj)

                if current_ddi_rate <= TARGET_DDI:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0.0, 1 + (TARGET_DDI - current_ddi_rate) / KP)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                
                del result, loss_ddi, loss_bce, loss_multi, loss
                del result_np, y_label

            llprint("\rtraining epoch {} step: {} / {}".format(epoch+1, step+1, len(data_train)))

        # 整体验估（用于选择最佳epoch）
        smm_record = []
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        
        model.eval()
        with torch.no_grad():
            for step, input in enumerate(data_eval):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
                for adm_idx, adm in enumerate(input):
                    target_output, _ = model(input[: adm_idx + 1])
                    y_gt_tmp = np.zeros(voc_size[2])
                    for item in adm[2]:
                        y_gt_tmp[item] = 1
                    y_gt.append(y_gt_tmp)
                    
                    target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
                    y_pred_prob.append(target_output)
                    
                    y_pred_tmp = target_output.copy()
                    y_pred_tmp[y_pred_tmp >= 0.5] = 1
                    y_pred_tmp[y_pred_tmp < 0.5] = 0
                    y_pred.append(y_pred_tmp)
                    
                    y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                    y_pred_label.append(sorted(y_pred_label_tmp))
                    visit_cnt += 1
                    med_cnt += len(y_pred_label_tmp)
                
                smm_record.append(y_pred_label)
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
                    np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
                )
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)
                llprint("\reval step: {} / {}".format(step, len(data_eval)))

        ddi_rate = calc_ddi_rate(smm_record, ddi_adj)
        metrics = {
            'ddi_rate': ddi_rate,
            'jaccard': np.mean(ja),
            'prauc': np.mean(prauc),
            'avg_f1': np.mean(avg_f1),
            'avg_med': med_cnt / visit_cnt if visit_cnt > 0 else 0,
        }

        ja_now = metrics['jaccard']
        if ja_now > best_ja:
            best_ja = ja_now
            best_metrics = metrics.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - tic
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d} | JA={ja_now:.4f} | F1={metrics['avg_f1']:.4f} "
                  f"| PRAUC={metrics['prauc']:.4f} | DDI={ddi_rate:.4f} "
                  f"| Med={metrics['avg_med']:.1f} | {elapsed:.1f}s")

    return model, best_model_state, best_metrics


# ============================================================
# 主实验流程
# ============================================================

def main():
    print("=" * 70)
    print("  Number of Visits Sensitivity Experiment for LLMDGTN")
    print("=" * 70)

    # 加载数据
    bundle = load_data()

    # 构建统计信息
    stats_path = '../data/mimic4_stats'
    count_conditional_prob_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_eval'], stats_path)

    # 结果存储
    output_dir = os.path.join("saved", "LLMDGTN", "num_visits_experiment")
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有seed的结果
    all_visit_results = {}

    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"  Run {run_idx + 1}/{len(SEEDS)} with seed={seed}")
        print(f"{'='*70}")

        # 训练模型
        model, best_state, best_metrics = train_and_evaluate(seed, bundle, epochs=EPOCH)
        
        # 加载最佳模型权重
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        print(f"\n  >>> Best overall JA={best_metrics['jaccard']:.4f}")
        print(f"  >>> Evaluating by visit groups...\n")

        # 按就诊次数分组评估
        visit_results = eval_model_by_visits(
            model, bundle['data_eval'], bundle['voc_size'], 
            bundle['ddi_adj'], visit_bins=VISIT_BINS
        )
        
        # 保存该run的结果
        all_visit_results[f'seed_{seed}'] = visit_results
        
        # 打印当前run的详细结果
        print(f"\n  {'Visit Group':<12} | {'Patients':>10} | {'Visits':>8} | "
              f"{'Jaccard':>10} | {'F1 Score':>10} | {'PRAUC':>10} | {'DDI Rate':>10} | {'Avg Med':>9}")
        print(f"  {'-'*100}")
        
        for label in VISIT_LABELS:
            if label in visit_results:
                r = visit_results[label]
                print(f"  {label:<12} | {r['patient_cnt']:>10} | {r['visit_cnt']:>8} | "
                      f"{r['jaccard']:>10.4f} | {r['avg_f1']:>10.4f} | {r['prauc']:>10.4f} | "
                      f"{r['ddi_rate']:>10.4f} | {r['avg_med']:>9.2f}")
        
        # 释放显存
        del model, best_state
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # 计算多轮平均结果
    # ================================================================
    
    print("\n" + "=" * 110)
    print(f"  AVERAGED RESULTS over {len(SEEDS)} runs (mean)")
    print("=" * 110)
    
    # 计算每个分组的平均指标
    averaged_results = {}
    for label in VISIT_LABELS:
        metrics_list = []
        for seed in SEEDS:
            if label in all_visit_results[f'seed_{seed}']:
                metrics_list.append(all_visit_results[f'seed_{seed}'][label])
        
        if len(metrics_list) > 0:
            averaged_results[label] = {
                'ddi_rate': np.mean([m['ddi_rate'] for m in metrics_list]),
                'jaccard': np.mean([m['jaccard'] for m in metrics_list]),
                'prauc': np.mean([m['prauc'] for m in metrics_list]),
                'avg_f1': np.mean([m['avg_f1'] for m in metrics_list]),
                'avg_med': np.mean([m['avg_med'] for m in metrics_list]),
                'patient_cnt': metrics_list[0]['patient_cnt'],
                'visit_cnt': metrics_list[0]['visit_cnt'],
            }
    
    # 打印平均结果表格
    header = (
        f"{'Visit Group':<12} | {'Patients':>10} | {'Visits':>8} | "
        f"{'Jaccard':>10} | {'F1 Score':>10} | {'PRAUC':>10} | {'DDI Rate':>10} | {'Avg Med':>9}"
    )
    print(f"\n  {header}")
    print(f"  {'-'*105}")
    
    rows = []
    for label in VISIT_LABELS:
        if label in averaged_results:
            r = averaged_results[label]
            row_str = (
                f"{label:<12} | {r['patient_cnt']:>10} | {r['visit_cnt']:>8} | "
                f"{r['jaccard']:>10.4f} | {r['avg_f1']:>10.4f} | {r['prauc']:>10.4f} | "
                f"{r['ddi_rate']:>10.4f} | {r['avg_med']:>9.2f}"
            )
            print(f"  {row_str}")
            rows.append(r)

    # ================================================================
    # 保存结果文件
    # ================================================================
    
    import csv
    
    # 1) CSV 完整结果
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Visit_Group', 'Patient_Count', 'Visit_Count',
            'DDI_Rate', 'Jaccard', 'PRAUC', 'AVG_F1', 'AVG_Med',
        ])
        for label in VISIT_LABELS:
            if label in averaged_results:
                r = averaged_results[label]
                writer.writerow([
                    label, r['patient_cnt'], r['visit_cnt'],
                    f"{r['ddi_rate']:.5f}", f"{r['jaccard']:.5f}",
                    f"{r['prauc']:.5f}", f"{r['avg_f1']:.5f}", f"{r['avg_med']:.4f}",
                ])
    print(f"\n  CSV saved: {csv_path}")

    # 2) 论文格式表格
    table_path = os.path.join(output_dir, "summary_table.txt")
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("Table: The effect of the number of visits for LLMDGTN on MIMIC-IV.\n\n")
        f.write(header + "\n")
        f.write("-" * 105 + "\n")
        for label in VISIT_LABELS:
            if label in averaged_results:
                r = averaged_results[label]
                row = (
                    f"{label:<12} | {r['patient_cnt']:>10} | {r['visit_cnt']:>8} | "
                    f"{r['jaccard']:>10.4f} | {r['avg_f1']:>10.4f} | {r['prauc']:>10.4f} | "
                    f"{r['ddi_rate']:>10.4f} | {r['avg_med']:>9.2f}"
                )
                f.write(row + "\n")
    print(f"  Table saved: {table_path}")

    # ================================================================
    # 绘图
    # ================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_labels = VISIT_LABELS
    x_pos = np.arange(len(x_labels))
    
    # 提取数据用于绑图
    jaccards = [averaged_results[l]['jaccard'] for l in x_labels if l in averaged_results]
    f1_scores = [averaged_results[l]['avg_f1'] for l in x_labels if l in averaged_results]
    praucs = [averaged_results[l]['prauc'] for l in x_labels if l in averaged_results]
    ddi_rates = [averaged_results[l]['ddi_rate'] for l in x_labels if l in averaged_results]
    avg_meds = [averaged_results[l]['avg_med'] for l in x_labels if l in averaged_results]
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548']
    
    # 子图1: Jaccard & F1
    ax1 = axes[0, 0]
    ax1.plot(x_pos, jaccards, 'o-', color=colors[0], linewidth=2, markersize=8, label='Jaccard')
    ax1.plot(x_pos, f1_scores, 's--', color=colors[1], linewidth=2, markersize=8, label='F1 Score')
    ax1.set_xlabel('Number of Visits', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('(a) Jaccard & F1 Score vs. Visits', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: PRAUC
    ax2 = axes[0, 1]
    ax2.plot(x_pos, praucs, '^-', color=colors[2], linewidth=2, markersize=8, label='PRAUC')
    ax2.set_xlabel('Number of Visits', fontsize=11)
    ax2.set_ylabel('PRAUC', fontsize=11)
    ax2.set_title('(b) PRAUC vs. Visits', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: DDI Rate
    ax3 = axes[1, 0]
    ax3.bar(x_pos, ddi_rates, color=colors[3], alpha=0.7, edgecolor='black')
    ax3.axhline(y=TARGET_DDI, color='red', linestyle='--', linewidth=1.5, label=f'Target DDI ({TARGET_DDI})')
    ax3.set_xlabel('Number of Visits', fontsize=11)
    ax3.set_ylabel('DDI Rate', fontsize=11)
    ax3.set_title('(c) DDI Rate vs. Visits', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: Avg Medications
    ax4 = axes[1, 1]
    ax4.plot(x_pos, avg_meds, 'D-', color=colors[4], linewidth=2, markersize=8, label='Avg # Drugs')
    ax4.fill_between(x_pos, avg_meds, alpha=0.2, color=colors[4])
    ax4.set_xlabel('Number of Visits', fontsize=11)
    ax4.set_ylabel('Average # of Drugs', fontsize=11)
    ax4.set_title('(d) Avg. # of Recommended Drugs vs. Visits', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, rotation=45, ha='right')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Fig3.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # 同时保存论文用的精简版图（只显示主要指标）
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_pos, jaccards, 'o-', color=colors[0], linewidth=2.5, markersize=10, label='Jaccard')
    ax.plot(x_pos, f1_scores, 's--', color=colors[1], linewidth=2.5, markersize=10, label='F1 Score')
    ax.plot(x_pos, praucs, '^-.', color=colors[2], linewidth=2.5, markersize=10, label='PRAUC')
    
    ax.set_xlabel('Number of Patient Visits', fontsize=13)
    ax.set_ylabel('Performance Score', fontsize=13)
    ax.set_title('Performance of LLMDGTN vs. Number of Visits on MIMIC-IV', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # 在每个点上标注数值
    for i, (ja, f1, pr) in enumerate(zip(jaccards, f1_scores, praucs)):
        if i % 2 == 0:  # 隔开标注避免拥挤
            ax.annotate(f'{ja:.3f}', (x_pos[i], ja), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color=colors[0])
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "Fig3_paper.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Paper figure saved: {fig2_path}")

    print("\n" + "=" * 70)
    print("  Experiment completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
