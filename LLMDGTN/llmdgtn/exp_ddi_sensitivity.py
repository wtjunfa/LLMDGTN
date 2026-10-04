r"""
DDI Rate Sensitivity (γ) Experiment
=====================================
Loop over different target DDI rate thresholds (γ), train and evaluate LLMDGTN model.

Reference: DNMDR paper Table 6 - "The performance of DNMDR with varying threshold γ"

Usage:
    cd to this directory
    python exp_ddi_sensitivity.py

Output:
    saved/LLMDGTN/ddi_sensitivity_experiment/
        - results.csv              full results per γ value
        - summary_table.txt       paper-style summary table
        - Fig_DDI_sensitivity.png visualization chart
"""

import os
import sys
import json
import pickle
import dill
import time
import gc
import csv
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# 确保能 import 同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LLMDGTN
from util import (
    llprint, multi_label_metric, ddi_rate_score,
    get_n_params, buildMPNN
)

# ============================================================
# 实验配置（优化版 - 半天内完成）
# ============================================================
SEEDS = [727, 2023, 3407]                           # 3 seeds -> 真实 mean±std（误差棒可见）
EPOCH = 50                                          # 与主结果一致的 epoch 数
GAMMA_LIST = [0.04, 0.06, 0.08]                    # 只测3个关键点（低/默认/高）
EMB_DIM = 64                                        # 固定 embedding 维度
LR = 5e-4                                           # 固定学习率
KP = 0.05
DDI_LAMBDA = 5.0                                    # DDI 正则强度（敏感性实验调大以展示 γ 权衡）

# 并行分工支持：--gammas 指定本进程只跑哪些 γ；--suffix 区分进度文件避免并行冲突
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--gammas', type=str, default=None, help="逗号分隔，如 0.04,0.06；默认跑全部")
_parser.add_argument('--suffix', type=str, default='', help='进度文件后缀，如 _A；非空时只训练不画图')
ARGS = _parser.parse_args()
RUN_GAMMAS = [float(g) for g in ARGS.gammas.split(',')] if ARGS.gammas else GAMMA_LIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running device: {device}")


# ============================================================
# 条件概率计算函数（来自 LLMDGTN.py）
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
                mp_cond_probs[pm] = 0.0

    pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
    pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'), -1)
    pickle.dump(med_probs, open(output_path + '/med_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_probs, open(output_path + '/dm_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_cond_probs, open(output_path + '/dm_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(md_cond_probs, open(output_path + '/md_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(pm_cond_probs, open(output_path + '/pm_probs.empirical.p', 'wb'), -1)
    pickle.dump(mp_cond_probs, open(output_path + '/mp_cond_probs.empirical.p', 'wb'), -1)


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
# 数据加载（与 LLMDGTN.py 一致，只加载一次）
# ============================================================

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mimic-iv")
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
        len(voc['diag_voc'].idx2word),
        len(voc['pro_voc'].idx2word),
        len(voc['med_voc'].idx2word)
    )
    print(f"Vocabulary size: diag={voc_size[0]}, pro={voc_size[1]}, med={voc_size[2]}")
    print(f"Train: {len(data_train)}, Eval: {len(data_eval)}, Test: {len(data_test)}")

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
# 评估函数
# ============================================================

def eval_model(model, data_eval, voc_size, ddi_adj):
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

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
            
            # 每50步或最后10步打印进度
            if step % 50 == 0 or step >= len(data_eval) - 10:
                llprint("\reval step: {} / {}".format(step + 1, len(data_eval)))

    print("\n[INFO] Calculating DDI rate...", flush=True)
    ddi_rate = calc_ddi_rate(smm_record, ddi_adj)
    return {
        'ddi_rate': ddi_rate,
        'jaccard': np.mean(ja),
        'prauc': np.mean(prauc),
        'avg_p': np.mean(avg_p),
        'avg_r': np.mean(avg_r),
        'avg_f1': np.mean(avg_f1),
        'avg_med': med_cnt / visit_cnt if visit_cnt > 0 else 0,
    }


# ============================================================
# 单次训练+评估（固定 emb_dim 和 lr，变化 target_ddi/γ）
# ============================================================

def train_and_evaluate(gamma, seed, data_bundle, emb_dim=EMB_DIM, epochs=EPOCH):
    """用指定的 γ (target_ddi) 和 seed 训练一轮，返回最佳 epoch 的指标"""
    # 清理显存
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

    # 构建模型
    model = LLMDGTN(
        voc_size, ehr_adj, ddi_adj, ddi_mask_H,
        MPNNSet, N_fingerprint, average_projection,
        emb_dim=emb_dim, device=device,
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    # 不使用混合精度/gradient checkpoint（与主模型 LLMDGTN.py 一致，速度更快）

    n_params = get_n_params(model)

    best_ja = 0
    best_metrics = None
    history = []

    print(f"\n{'='*60}")
    print(f"  γ={gamma:.2f}, seed={seed}, dim={emb_dim}, params={n_params:,}")
    print(f"{'='*60}")

    def _model_forward(seq_input):
        return model(seq_input)

    for epoch in range(epochs):
        tic = time.time()
        model.train()

        # ---- 训练 ----
        for step, input in enumerate(data_train):
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
                for idx, item in enumerate(adm[2]):
                    safe_item = max(0, min(int(item), voc_size[2] - 1))
                    loss_multi_target[0][idx] = safe_item

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)
                )

                result_np = F.sigmoid(result).detach().cpu().numpy()[0]
                result_np[result_np >= 0.5] = 1
                result_np[result_np < 0.5] = 0
                y_label = np.where(result_np == 1)[0]
                current_ddi_rate = calc_ddi_rate([[y_label]], ddi_adj)

                # 优化后 loss：BCE 主导 + DDI 温和正则（γ=target_ddi 控制惩罚触发阈值）
                loss_base = loss_bce + 0.05 * loss_multi
                ddi_slack = max(0.0, current_ddi_rate - gamma) / gamma
                loss = loss_base + DDI_LAMBDA * (1.0 + ddi_slack) * loss_ddi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                del result, loss_ddi, loss_bce, loss_multi, loss
                del result_np, y_label

            if (step + 1) % 500 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            llprint("\rtraining epoch {} step: {} / {}".format(epoch+1, step+1, len(data_train)))

        # ---- 验证 ----
        metrics = eval_model(model, data_eval, voc_size, ddi_adj)
        history.append(metrics)
        elapsed = time.time() - tic

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        ja_now = metrics['jaccard']
        if ja_now > best_ja:
            best_ja = ja_now
            best_metrics = metrics.copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d} | JA={ja_now:.4f} | F1={metrics['avg_f1']:.4f} "
                  f"| PRAUC={metrics['prauc']:.4f} | DDI={metrics['ddi_rate']:.4f} "
                  f"| Med={metrics['avg_med']:.1f} | {elapsed:.1f}s")

    return best_metrics, n_params


# ============================================================
# 主实验流程
# ============================================================

def load_existing_results(output_dir):
    """加载已有的实验结果"""
    result_path = os.path.join(output_dir, f"experiment_progress{ARGS.suffix}.pkl")
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    return None


def save_experiment_progress(all_results, output_dir):
    """保存实验进度"""
    result_path = os.path.join(output_dir, f"experiment_progress{ARGS.suffix}.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(all_results, f)


def main():
    print("=" * 70)
    print("  DDI Rate Sensitivity (γ) Experiment for LLMDGTN")
    print("=" * 70)
    print(f"  Fixed emb_dim = {EMB_DIM}, LR = {LR}")
    print(f"  Testing γ (target DDI) values: {GAMMA_LIST}")

    # 加载数据
    bundle = load_data()

    # 构建统计信息（只需一次）
    stats_path = '../data/mimic4_stats'
    count_conditional_prob_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_eval'], stats_path)

    # 结果存储
    output_dir = os.path.join("saved", "LLMDGTN", "ddi_sensitivity_experiment")
    os.makedirs(output_dir, exist_ok=True)

    # 尝试加载已有结果（断点续跑）
    all_results = load_existing_results(output_dir) or {}

    for gamma in RUN_GAMMAS:
        gamma_key = f"gamma_{gamma:.2f}"

        if gamma_key not in all_results:
            all_results[gamma_key] = {'gamma': gamma, 'runs': []}

        completed_runs = len(all_results[gamma_key]['runs'])
        total_runs = len(SEEDS)

        if completed_runs >= total_runs:
            print(f"\n{'='*70}")
            print(f"  [SKIP] gamma={gamma:.2f} completed ({completed_runs}/{total_runs} runs)")
            print(f"{'='*70}")
            continue

        print(f"\n{'='*70}")
        print(f"  [START] gamma={gamma:.2f} (completed {completed_runs}/{total_runs})")
        print(f"{'='*70}")

        run_metrics_list = []

        for existing_run in all_results[gamma_key]['runs']:
            run_metrics_list.append({
                k: v for k, v in existing_run.items()
                if k != 'seed' and k != 'params'
            })

        for run_idx in range(completed_runs, total_runs):
            seed = SEEDS[run_idx]

            torch.cuda.empty_cache()
            gc.collect()

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[GPU Memory] Before run {run_idx+1}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            best_metrics, n_params = train_and_evaluate(
                gamma=gamma, seed=seed, data_bundle=bundle, epochs=EPOCH
            )

            all_results[gamma_key]['runs'].append({
                'seed': seed,
                **best_metrics,
                'params': n_params,
            })

            run_metrics_list.append(best_metrics)

            save_experiment_progress(all_results, output_dir)

            print(f"\n  >>> γ={gamma:.2f}, run {run_idx+1}/{total_runs} done | "
                  f"best JA={best_metrics['jaccard']:.4f}\n")

            del best_metrics
            torch.cuda.empty_cache()
            gc.collect()

        # 计算 mean ± std
        keys_of_interest = ['ddi_rate', 'jaccard', 'prauc', 'avg_f1', 'avg_med', 'avg_p', 'avg_r']
        summary = {'gamma': gamma, 'params': n_params}
        for k in keys_of_interest:
            vals = [r[k] for r in run_metrics_list]
            summary[f'{k}_mean'] = np.mean(vals)
            summary[f'{k}_std'] = np.std(vals)
        all_results[gamma_key]['summary'] = summary

        save_experiment_progress(all_results, output_dir)

    # ================================================================
    # 输出结果表格
    # ================================================================

    print("\n" + "=" * 120)
    print("  RESULTS SUMMARY (mean ± std over {} runs)".format(len(SEEDS)))
    print("=" * 120)

    header = (
        f"{'γ':^6} | {'DDI Rate':^22} | {'Jaccard':^22} | {'F1 Score':^22} | "
        f"{'PRAUC':^22} | {'Avg#Drugs':^18}"
    )
    print(header)
    print("-" * 130)

    rows = []
    for gamma in GAMMA_LIST:
        gamma_key = f'gamma_{gamma:.2f}'
        
        # 检查是否有summary，如果没有则计算
        if 'summary' not in all_results[gamma_key]:
            run_metrics_list = all_results[gamma_key].get('runs', [])
            if len(run_metrics_list) >= 1:  # 至少有1个run就计算
                keys_of_interest = ['ddi_rate', 'jaccard', 'prauc', 'avg_f1', 'avg_med', 'avg_p', 'avg_r']
                summary = {'gamma': gamma}
                for k in keys_of_interest:
                    vals = [r[k] for r in run_metrics_list]
                    summary[f'{k}_mean'] = np.mean(vals)
                    summary[f'{k}_std'] = np.std(vals)
                all_results[gamma_key]['summary'] = summary
            else:
                print(f"  [WARN] No completed runs for γ={gamma:.2f}, skipping...")
                continue
        
        s = all_results[gamma_key]['summary']
        row = (
            f"{gamma:^6.2f} | "
            f"{s['ddi_rate_mean']:.5f} +/- {s['ddi_rate_std']:.5f}{'':<8} | "
            f"{s['jaccard_mean']:.4f} +/- {s['jaccard_std']:.4f}{'':<8} | "
            f"{s['avg_f1_mean']:.4f} +/- {s['avg_f1_std']:.4f}{'':<8} | "
            f"{s['prauc_mean']:.4f} +/- {s['prauc_std']:.4f}{'':<8} | "
            f"{s['avg_med_mean']:.2f} +/- {s['avg_med_std']:.2f}{'':<10}"
        )
        print(row)
        rows.append(s)

    print("-" * 130)
    
    if rows:
        best_gamma = max(GAMMA_LIST, 
                        key=lambda g: f'gamma_{g:.2f}' in all_results and 
                                       'summary' in all_results[f'gamma_{g:.2f}'] and
                                       all_results[f'gamma_{g:.2f}']['summary']['jaccard_mean'])
        print(f"\n  >> Best γ by Jaccard: **{best_gamma:.2f}**")
        print(f"     (Jaccard={all_results[f'gamma_{best_gamma:.2f}']['summary']['jaccard_mean']:.4f})")

    # 并行分工模式：只训练并保存进度，跳过汇总输出（等全部进程完成后由主模式汇总画图）
    if ARGS.suffix:
        print(f"  [MODE suffix={ARGS.suffix}] 训练完成，跳过 CSV/LaTeX/图输出")
        return

    # ================================================================
    # 保存到文件
    # ================================================================

    # 1) CSV
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Gamma', 'Params',
            'DDI_Rate_mean', 'DDI_Rate_std',
            'Jaccard_mean', 'Jaccard_std',
            'PRAUC_mean', 'PRAUC_std',
            'AVG_F1_mean', 'AVG_F1_std',
            'AVG_Med_mean', 'AVG_Med_std',
        ])
        for gamma in GAMMA_LIST:
            gamma_key = f'gamma_{gamma:.2f}'
            if 'summary' in all_results[gamma_key]:
                s = all_results[gamma_key]['summary']
                writer.writerow([
                    f"{gamma:.2f}", s.get('params', ''),
                    f"{s['ddi_rate_mean']:.5f}", f"{s['ddi_rate_std']:.5f}",
                    f"{s['jaccard_mean']:.5f}", f"{s['jaccard_std']:.5f}",
                    f"{s['prauc_mean']:.5f}", f"{s['prauc_std']:.5f}",
                    f"{s['avg_f1_mean']:.5f}", f"{s['avg_f1_std']:.5f}",
                    f"{s['avg_med_mean']:.4f}", f"{s['avg_med_std']:.4f}",
                ])
    print(f"\n  CSV saved: {csv_path}")

    # 2) 论文格式表格 (LaTeX)
    table_path = os.path.join(output_dir, "summary_table.txt")
    latex_content = []
    latex_content.append("% Table: The performance of LLMDGTN with varying threshold $\\gamma$.")
    latex_content.append("\\begin{table}[t]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{The performance of LLMDGTN with varying threshold $\\gamma$.}")
    latex_content.append("\\label{tab:ddi_sensitivity}")
    latex_content.append("\\adjustbox{max width=\\linewidth}{")
    latex_content.append("\\small")
    latex_content.append("\\begin{tabular}{cccccc}")
    latex_content.append("\\toprule")
    latex_content.append("$\\gamma$ & DDI Rate & Jaccard & F1 & PRAUC & Avg.\\ \\# \\\\")
    latex_content.append("\\midrule")
    
    for gamma in GAMMA_LIST:
        gamma_key = f'gamma_{gamma:.2f}'
        if 'summary' not in all_results[gamma_key]:
            continue
        s = all_results[gamma_key]['summary']
        latex_content.append(
            f"{gamma:.2f} & "
            f"{s['ddi_rate_mean']:.5f} & "
            f"{s['jaccard_mean']:.4f} & "
            f"{s['avg_f1_mean']:.4f} & "
            f"{s['prauc_mean']:.4f} & "
            f"{s['avg_med_mean']:.2f} \\\\"
        )

    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}}")
    latex_content.append("\\end{table}")
    
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    print(f"  LaTeX Table saved: {table_path}")

    # 3) 可视化图
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.size'] = 12

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('Effect of Acceptable DDI Rate ($\\gamma$) on LLMDGTN Performance', fontsize=14, fontweight='bold')

        gammas = GAMMA_LIST
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

        metrics_config = [
            ('ddi_rate', 'DDI Rate ↓', axes[0, 0]),
            ('jaccard', 'Jaccard ↑', axes[0, 1]),
            ('avg_f1', 'F1 Score ↑', axes[0, 2]),
            ('prauc', 'PRAUC ↑', axes[1, 0]),
            ('avg_med', 'Avg #Drugs', axes[1, 1]),
            ('avg_r', 'Recall ↑', axes[1, 2]),
        ]

        for metric_name, title, ax in metrics_config:
            # 只使用有summary的gamma值
            valid_gammas = [g for g in gammas if 'summary' in all_results[f'gamma_{g:.2f}']]
            means = [all_results[f'gamma_{g:.2f}']['summary'][f'{metric_name}_mean'] for g in valid_gammas]
            stds = [all_results[f'gamma_{g:.2f}']['summary'][f'{metric_name}_std'] for g in valid_gammas]

            bars = ax.bar(range(len(valid_gammas)), means, yerr=stds, capsize=5,
                         color='#3498DB', edgecolor='#2980B9', linewidth=1.5, alpha=0.85)
            
            # 收窄纵坐标：数据范围上下各留 padding，让柱子差异更明显、更美观
            lo = float(np.min(means) - np.max(stds))
            hi = float(np.max(means) + np.max(stds))
            span = hi - lo if hi - lo > 0 else abs(hi) * 0.01 + 1e-6
            pad = span * 0.6
            ax.set_ylim(lo - pad, hi + pad)
            
            ax.set_xticks(range(len(valid_gammas)))
            ax.set_xticklabels([f'{g:.2f}' for g in valid_gammas])
            ax.set_xlabel('$\\gamma$ (Target DDI)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds) * 0.1,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig_path = os.path.join(output_dir, "Fig_DDI_sensitivity.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure saved: {fig_path}")

    except ImportError:
        print("  [Warning] matplotlib not installed, skipping figure generation.")

    print("\nDone!")


if __name__ == "__main__":
    main()
