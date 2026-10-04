r"""
Multi-Model Number of Visits Sensitivity Experiment
=====================================================
Compare LR, GAMENet, SafeDrug, and LLMDGTN performance across different visit counts (1-10).

Usage:
    cd to this directory
    python exp_multi_model_visits.py

Output:
    saved/LLMDGTN/multi_model_visits/
        - results.csv          full results per model per visit count
        - summary_table.txt    paper-style summary table
        - Fig4.png             multi-model comparison plot (4 subplots)
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
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LLMDGTN
from util import (
    llprint, multi_label_metric, ddi_rate_score,
    get_n_params, buildMPNN
)

# 导入GAMENet和SafeDrug（从checkpoint models）
import importlib.util
def load_model_from_checkpoint(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

checkpoint_models = load_model_from_checkpoint("checkpoint_models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ipynb_checkpoints", "models-checkpoint.py"))
GAMENet = checkpoint_models.GAMENet
SafeDrugModel = checkpoint_models.SafeDrugModel

# ============================================================
# 配置 - 快速版本
# ============================================================
SEEDS = [1208]  # 1次实验（快速版）
EPOCH = 20  # 减少epochs加速
TARGET_DDI = 0.06
KP = 0.05
LR_RATE = 5e-4
EMB_DIM = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running device: {device}")

# 就诊次数: 1, 2, 3, ..., 10
VISIT_COUNTS = list(range(1, 11))
MODELS_TO_TEST = ['LR', 'GAMENet', 'SafeDrug', 'LLMDGTN']

# 模型颜色配置
MODEL_COLORS = {
    'LR': '#1f77b4',
    'GAMENet': '#ff7f0e', 
    'SafeDrug': '#2ca02c',
    'LLMDGTN': '#d62728'
}
MODEL_MARKERS = {
    'LR': 'o',
    'GAMENet': 's',
    'SafeDrug': '^',
    'LLMDGTN': 'D'
}


# ============================================================
# 数据辅助函数
# ============================================================

def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
    dx_freqs, proc_freqs, med_freqs = {}, {}, {}
    dm_freqs, pm_freqs = {}, {}
    total_visit = 0
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()
        if train_key_set is not None and seqex not in train_key_set:
            total_visit += len(seqex)
            continue
        for key in seqex:
            dx_ids, proc_ids, med_ids = key[0], key[1], key[2]
            for dx in dx_ids:
                dx_freqs[dx] = dx_freqs.get(dx, 0) + 1
            for proc in proc_ids:
                proc_freqs[proc] = proc_freqs.get(proc, 0) + 1
            for med in med_ids:
                med_freqs[med] = med_freqs.get(med, 0) + 1
            for dx in dx_ids:
                for med in med_ids:
                    dm = str(dx) + ',' + str(med)
                    dm_freqs[dm] = dm_freqs.get(dm, 0) + 1
            for proc in proc_ids:
                for med in med_ids:
                    pm = str(proc) + ',' + str(med)
                    pm_freqs[pm] = pm_freqs.get(pm, 0) + 1
            total_visit += 1

    n = float(total_visit) if total_visit > 0 else 1.0
    dx_probs = {k: v/n for k, v in dx_freqs.items()}
    proc_probs = {k: v/n for k, v in proc_freqs.items()}
    med_probs = {k: v/n for k, v in med_freqs.items()}
    dm_probs = {k: v/n for k, v in dm_freqs.items()}
    pm_probs = {k: v/n for k, v in pm_freqs.items()}

    dm_cond, md_cond = {}, {}
    for dx, dp in dx_probs.items():
        for med, mp in med_probs.items():
            dm, md = str(dx)+','+str(med), str(med)+','+str(dx)
            dm_cond[dm] = dm_probs[dm]/dp if dm in dm_probs else 0.0
            md_cond[md] = dm_probs[dm]/mp if dm in dm_probs else 0.0

    pm_cond, mp_cond = {}, {}
    for pp, pp_prob in proc_probs.items():
        for med, mp in med_probs.items():
            pm, mmp = str(pp)+','+str(med), str(med)+','+str(pp)
            pm_cond[pm] = pm_probs[pm]/pp_prob if pm in pm_probs else 0.0
            mp_cond[mmp] = pm_probs[pm]/mp if pm in pm_probs else 0.0

    for name, data in [('dx_probs', dx_probs), ('proc_probs', proc_probs), ('med_probs', med_probs),
                        ('dm_probs', dm_probs), ('dm_cond_probs', dm_cond), ('md_cond_probs', md_cond),
                        ('pm_probs', pm_probs), ('pm_cond_probs', pm_cond), ('mp_cond_probs', mp_cond)]:
        pickle.dump(data, open(output_path + f'/{name}.empirical.p', 'wb'), -1)


def add_sparse_prior_guide_dp(seqex_list, stats_path, key_set=None):
    print('Loading conditional probabilities.')
    dm_cond = pickle.load(open(stats_path + '/dm_cond_probs.empirical.p', 'rb'))
    md_cond = pickle.load(open(stats_path + '/md_cond_probs.empirical.p', 'rb'))
    pm_cond = pickle.load(open(stats_path + '/pm_cond_probs.empirical.p', 'rb'))
    mp_cond = pickle.load(open(stats_path + '/mp_cond_probs.empirical.p', 'rb'))
    
    print('Adding prior guide.')
    for seqex in seqex_list:
        for key in seqex:
            dx_ids, proc_idx, med_ids = key[0], key[1], key[2]
            indices, values = [], []
            for i, dx in enumerate(dx_ids):
                for j, med in enumerate(med_ids):
                    indices.append((i, len(dx_ids)+len(proc_idx)+j))
                    values.append(dm_cond.get(str(dx)+','+str(med), 0.0))
            for i, proc in enumerate(proc_idx):
                for j, med in enumerate(med_ids):
                    indices.append((len(dx_ids)+i, len(dx_ids)+len(proc_idx)+j))
                    values.append(pm_cond.get(str(proc)+','+str(med), 0.0))
            for i, med in enumerate(med_ids):
                for j, dx in enumerate(dx_ids):
                    indices.append((len(dx_ids)+len(proc_idx)+med, j))
                    values.append(md_cond.get(str(med)+','+str(dx), 0.0))
                for j, proc in enumerate(proc_idx):
                    indices.append((len(dx_ids)+len(proc_idx)+med, len(dx_ids)+j))
                    values.append(mp_cond.get(str(med)+','+str(proc), 0.0))
            key.append(indices)
            key.append(values)


def load_data():
    """加载数据集"""
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
        return json.load(open(path, 'r', encoding='utf-8'))

    voc = load_vocab(voc_path)

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    med_map = {i: code for i, code in enumerate(molecule.keys())}
    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_map, 2, device)

    voc_size = (len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word']))
    print(f"Vocab: diag={voc_size[0]}, pro={voc_size[1]}, med={voc_size[2]}")
    print(f"Train: {len(data_train)}, Eval: {len(data_eval)}, Test: {len(data_test)}")

    # 统计各就诊次数的患者数量
    visit_dist = defaultdict(int)
    for patient in data_eval:
        n = min(len(patient), 15)  # 截断到15
        visit_dist[n] += 1
    print("\nEval set visit distribution:")
    for v in sorted(visit_dist.keys()):
        if v <= 12:
            print(f"  {v} visits: {visit_dist[v]} patients")

    return {
        'ehr_adj': ehr_adj, 'ddi_adj': ddi_adj, 'ddi_mask_H': ddi_mask_H,
        'data_train': data_train, 'data_eval': data_eval, 'data_test': data_test,
        'MPNNSet': MPNNSet, 'N_fingerprint': N_fingerprint,
        'average_projection': average_projection, 'voc_size': voc_size,
        'ddi_adj_matrix': ddi_adj,
    }


def calc_ddi_rate(pred_records, ddi_mat):
    ddi_cnt, pair_cnt = 0, 0
    for patient in pred_records:
        for meds in patient:
            meds = list(set(meds))
            for i in range(len(meds)):
                for j in range(i+1, len(meds)):
                    pair_cnt += 1
                    if ddi_mat[meds[i], meds[j]] == 1:
                        ddi_cnt += 1
    return ddi_cnt / pair_cnt if pair_cnt > 0 else 0.0


# ============================================================
# 按精确就诊次数评估（每个visit数单独计算）
# ============================================================

def evaluate_by_exact_visits(model, data_eval, voc_size, ddi_adj, model_name='unknown'):
    """
    按精确就诊次数分组评估
    
    Returns:
        dict: {visit_num: {metrics...}}
    """
    is_pytorch = hasattr(model, 'eval')
    if is_pytorch:
        model.eval()
    
    results = {}
    
    for target_visits in VISIT_COUNTS:
        ja_list, prauc_list, f1_list = [], [], []
        smm_record = []
        med_cnt, visit_cnt = 0, 0
        
        with torch.no_grad() if is_pytorch else contextlib.nullcontext():
            for step, input in enumerate(data_eval):
                n_visits = len(input)
                
                if n_visits != target_visits:
                    continue
                
                y_gt, y_pred, y_pred_prob = [], [], []
                
                for adm_idx, adm in enumerate(input):
                    try:
                        if is_pytorch:
                            output = model(input[:adm_idx+1])
                            if isinstance(output, tuple):
                                target_output = output[0]
                            else:
                                target_output = output
                            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
                        else:
                            # LR模型
                            feat = extract_lr_features(input[:adm_idx+1], voc_size)
                            target_output = model.predict_proba(feat)
                            # 确保是1D数组
                            if target_output.ndim > 1:
                                target_output = target_output[0]
                        
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
                    except Exception as e:
                        continue
                
                if len(y_gt) > 0:
                    adm_ja, adm_prauc, _, _, adm_f1 = multi_label_metric(
                        np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
                    )
                    ja_list.append(adm_ja)
                    prauc_list.append(adm_prauc)
                    f1_list.append(adm_f1)
        
        if visit_cnt > 0:
            ddi_rate = calc_ddi_rate(smm_record, ddi_adj)
            results[target_visits] = {
                'jaccard': np.mean(ja_list) if ja_list else 0,
                'prauc': np.mean(prauc_list) if prauc_list else 0,
                'avg_f1': np.mean(f1_list) if f1_list else 0,
                'ddi_rate': ddi_rate,
                'avg_med': med_cnt / visit_cnt,
                'patient_cnt': 0,  # 后面填充
                'visit_cnt': visit_cnt,
            }
        else:
            results[target_visits] = {
                'jaccard': 0, 'prauc': 0, 'avg_f1': 0,
                'ddi_rate': 0, 'avg_med': 0,
                'patient_cnt': 0, 'visit_cnt': 0,
            }
        
        llprint(f"\r{model_name} visits={target_visits}: done ({visit_cnt} visits)")
    
    # 填充patient计数
    for target_visits in VISIT_COUNTS:
        cnt = sum(1 for p in data_eval if len(p) == target_visits)
        results[target_visits]['patient_cnt'] = cnt
    
    return results


import contextlib

# ============================================================
# LR特征提取和模型
# ============================================================

def extract_lr_features(patient_history, voc_size):
    """
    为LR模型提取特征向量
    基于患者历史诊断、程序、药物的多-hot编码统计
    """
    diag_dim, pro_dim, med_dim = voc_size
    
    features = []
    all_diags, all_procs, all_meds = [], [], []
    
    for adm in patient_history[:-1]:  # 历史记录
        all_diags.extend(adm[0])
        all_procs.extend(adm[1])
        all_meds.extend(adm[2])
    
    # 当前诊断和程序
    current_adm = patient_history[-1]
    current_diags = current_adm[0]
    current_procs = current_adm[1]
    
    # 特征：历史频率编码 + 当前状态
    from collections import Counter
    diag_counter = Counter(all_diags)
    proc_counter = Counter(all_procs)
    med_counter = Counter(all_meds)
    
    # 历史诊断one-hot (归一化)
    hist_diag = np.zeros(diag_dim)
    for d, c in diag_counter.items():
        if 0 <= d < diag_dim:
            hist_diag[d] = min(c / max(len(all_diags), 1), 1.0)
    
    # 历史程序one-hot
    hist_proc = np.zeros(pro_dim)
    for p, c in proc_counter.items():
        if 0 <= p < pro_dim:
            hist_proc[p] = min(c / max(len(all_procs), 1), 1.0)
    
    # 历史药物one-hot
    hist_med = np.zeros(med_dim)
    for m, c in med_counter.items():
        if 0 <= m < med_dim:
            hist_med[m] = min(c / max(len(all_meds), 1), 1.0)
    
    # 当前诊断one-hot
    cur_diag = np.zeros(diag_dim)
    for d in current_diags:
        if 0 <= d < diag_dim:
            cur_diag[d] = 1.0
    
    # 当前程序one-hot  
    cur_proc = np.zeros(pro_dim)
    for p in current_procs:
        if 0 <= p < pro_dim:
            cur_proc[p] = 1.0
    
    # 统计特征
    n_visits = len(patient_history)
    n_unique_meds = len(set(all_meds))
    avg_meds_per_visit = len(all_meds) / max(n_visits - 1, 1)
    
    # 拼接特征 (使用降维后的版本以避免维度灾难)
    # 使用稀疏特征的统计量代替完整one-hot
    features = [
        len(current_diags) / max(diag_dim, 1),      # 当前诊断密度
        len(current_procs) / max(pro_dim, 1),       # 当前程序密度
        len(set(current_diags)) / max(len(current_diags), 1),  # 诊断多样性
        len(set(current_procs)) / max(len(current_procs), 1),  # 程序多样性
        n_visits / 30.0,                             # 归一化就诊次数
        n_unique_meds / max(med_dim, 1),             # 药物多样性
        avg_meds_per_visit / 50.0,                   # 平均每次用药数
        hist_diag.sum() / max(hist_diag.size, 1),    # 历史诊断覆盖度
        hist_med.sum() / max(hist_med.size, 1),      # 历史药物覆盖度
        # 取top-k维度的one-hot作为特征
    ]
    
    # 添加最重要的药物特征 (top 100 most frequent meds)
    top_k = min(100, med_dim)
    features.extend(hist_med[:top_k].tolist())
    features.extend(cur_diag[:top_k].tolist())
    
    return np.array(features, dtype=np.float32)


class LRModel:
    """逻辑回归模型封装"""
    def __init__(self, voc_size):
        self.voc_size = voc_size
        self.model = MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=500, solver='lbfgs'), n_jobs=-1)
        self.fitted = False
    
    def train(self, data_train, voc_size):
        """训练LR模型"""
        print("Training Logistic Regression model...")
        X_train, y_train = [], []
        
        for patient in data_train:
            for idx, adm in enumerate(patient):
                if idx == 0:  # 第一次就诊没有历史，跳过
                    continue
                try:
                    feat = extract_lr_features(patient[:idx+1], voc_size)
                    X_train.append(feat)
                    
                    label = np.zeros(voc_size[2])
                    for item in adm[2]:
                        if 0 <= int(item) < voc_size[2]:
                            label[int(item)] = 1
                    y_train.append(label)
                except:
                    continue
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # 过滤掉标签全是0的列（没有正样本的药物）
        col_sums = y_train.sum(axis=0)
        valid_cols = col_sums > 10  # 至少10个正样本才保留
        y_train_filtered = y_train[:, valid_cols]
        
        print(f"  Training samples: {len(X_train)}, Feature dim: {X_train.shape[1]}")
        print(f"  Valid drug columns: {valid_cols.sum()} / {voc_size[2]}")
        
        if y_train_filtered.shape[1] == 0:
            print("  WARNING: No valid drug columns! Using all columns with small regularization")
            self.model = MultiOutputClassifier(
                LogisticRegression(C=100.0, max_iter=200, solver='liblinear'), n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train_filtered)
        self.fitted = True
        self.valid_cols = valid_cols
        
        print("  LR training completed!")
    
    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Model not fitted!")
        
        probs = self.model.predict_proba(X.reshape(1, -1))
        result = np.zeros(self.voc_size[2])  # 全长向量
        
        if hasattr(self, 'valid_cols'):
            for i, is_valid in enumerate(self.valid_cols):
                if is_valid and i < len(probs):
                    p = probs[i]
                    result[i] = p[0, 1] if p.shape[1] > 1 else p[0, 0]
        else:
            for i, p in enumerate(probs):
                if i < self.voc_size[2]:
                    result[i] = p[0, 1] if p.shape[1] > 1 else p[0, 0]
        
        return result


# ============================================================
# 训练函数
# ============================================================

def train_gamenet(seed, bundle, epochs=EPOCH):
    """训练GAMENet模型"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 使用checkpoint版本的GAMENet（已包含兼容的GCN）
    model = GAMENet(
        bundle['voc_size'], bundle['ehr_adj'], bundle['ddi_adj'],
        emb_dim=EMB_DIM, device=device, ddi_in_memory=True
    )
    
    # 修复GCN forward问题: checkpoint版本GCN需要(x, adj)
    # 但GAMENet调用的是self.ehr_gcn()无参，所以需要monkey-patch
    original_ehr_gcn_forward = model.ehr_gcn.forward
    original_ddi_gcn_forward = model.ddi_gcn.forward
    
    def patched_ehr_gcn_forward(self_ref):
        return original_ehr_gcn_forward(self_ref.x, self_ref.adj)
    
    def patched_ddi_gcn_forward(self_ref):
        return original_ddi_gcn_forward(self_ref.x, self_ref.adj)
    
    import types
    model.ehr_gcn.forward = types.MethodType(patched_ehr_gcn_forward, model.ehr_gcn)
    model.ddi_gcn.forward = types.MethodType(patched_ddi_gcn_forward, model.ddi_gcn)
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)  # GAMENet用较低学习率
    
    print(f"\n{'='*50}\n  Training GAMENet (seed={seed})\n{'='*50}")
    
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for step, input in enumerate(bundle['data_train']):
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                
                loss_bce_target = np.zeros((1, bundle['voc_size'][2]))
                loss_bce_target[:, adm[2]] = 1
                
                loss_multi_target = np.full((1, bundle['voc_size'][2]), -1)
                for item in adm[2]:
                    safe_item = max(0, min(int(item), bundle['voc_size'][2]-1))
                    loss_multi_target[0][safe_item] = 1
                
                target_output, loss_ddi = model(seq_input)
                
                loss_bce = F.binary_cross_entropy_with_logits(
                    target_output, torch.FloatTensor(loss_bce_target).to(device)
                )
                loss_multi = F.multilabel_margin_loss(
                    F.sigmoid(target_output), torch.LongTensor(loss_multi_target).to(device)
                )
                
                result_np = F.sigmoid(target_output).detach().cpu().numpy()[0]
                result_np[result_np >= 0.5] = 1
                result_np[result_np < 0.5] = 0
                y_label = np.where(result_np == 1)[0]
                current_ddi = calc_ddi_rate([[y_label]], bundle['ddi_adj'])
                
                if current_ddi <= TARGET_DDI:
                    loss = 0.9 * loss_bce + 0.1 * loss_multi
                else:
                    rnd = np.exp((TARGET_DDI - current_ddi) / 2.0)
                    if np.random.rand() < rnd:
                        loss = loss_ddi
                    else:
                        loss = 0.9 * loss_bce + 0.1 * loss_multi
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
            llprint(f"\rGAMENet epoch {epoch+1}/{epochs}: {step+1}/{len(bundle['data_train'])}")
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} done")
        
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def train_safedrug(seed, bundle, epochs=EPOCH):
    """训练SafeDrug模型（带维度自动修复）"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_drugs = bundle['voc_size'][2]
    
    # ========== 关键修复：确保所有数据维度匹配 ==========
    # 1. 修复 average_projection: 必须为 (n_drugs, mpnn_dim)
    avg_proj = bundle['average_projection'].clone()
    print(f"  Original average_projection shape: {avg_proj.shape}")
    
    if avg_proj.shape[0] != n_drugs:
        if avg_proj.shape[0] < n_drugs:
            # 用零填充到正确大小
            pad = torch.zeros(n_drugs - avg_proj.shape[0], avg_proj.shape[1])
            avg_proj = torch.cat([avg_proj, pad], dim=0)
        else:
            avg_proj = avg_proj[:n_drugs]
        print(f"  Fixed average_projection -> {avg_proj.shape}")
    
    # 2. 修复 ddi_mask_H: 确保与药物数一致
    ddi_mask_H = bundle['ddi_mask_H'].copy() if hasattr(bundle['ddi_mask_H'], 'copy') else bundle['ddi_mask_H']
    ddi_mask_H = np.array(ddi_mask_H)
    print(f"  Original ddi_mask_H shape: {ddi_mask_H.shape}")
    
    # ddi_mask_H 应该是 (n_drugs, feature_dim) 或类似的
    if ddi_mask_H.shape[0] != n_drugs:
        if ddi_mask_H.ndim == 2:
            if ddi_mask_H.shape[0] < n_drugs:
                pad = np.zeros((n_drugs - ddi_mask_H.shape[0], ddi_mask_H.shape[1]))
                ddi_mask_H = np.vstack([ddi_mask_H, pad])
            else:
                ddi_mask_H = ddi_mask_H[:n_drugs]
        print(f"  Fixed ddi_mask_H -> {ddi_mask_H.shape}")
    
    # 创建修复后的bundle副本
    fixed_bundle = {
        **bundle,
        'average_projection': avg_proj,
        'ddi_mask_H': ddi_mask_H
    }
    
    model = SafeDrugModel(
        fixed_bundle['voc_size'], fixed_bundle['ehr_adj'], fixed_bundle['ddi_adj'],
        fixed_bundle['ddi_mask_H'], fixed_bundle['MPNNSet'], fixed_bundle['N_fingerprint'],
        fixed_bundle['average_projection'], emb_dim=EMB_DIM, device=device
    )
    
    # 验证并最终修复MPNN层
    mpnn_dim = model.MPNN_emb.shape[0]
    print(f"  MPNN_emb output dim: {mpnn_dim}, expected drugs: {n_drugs}")
    
    if mpnn_dim != n_drugs:
        # 强制重建MPNN相关层
        model.MPNN_output = nn.Linear(n_drugs, n_drugs).to(device)
        model.MPNN_layernorm = nn.LayerNorm(n_drugs).to(device)
        
        # 调整MPNN_emb
        if mpnn_dim < n_drugs:
            pad = torch.zeros(n_drugs - mpnn_dim, model.MPNN_emb.shape[1]).to(device)
            model.MPNN_emb = torch.cat([model.MPNN_emb, pad], dim=0)
        else:
            model.MPNN_emb = model.MPNN_emb[:n_drugs]
        print(f"  Force-fixed MPNN layers to {n_drugs} dims")
    
    # 同样修复bipartite层
    bip_feat_dim = ddi_mask_H.shape[1]
    model.bipartite_transform = nn.Sequential(nn.Linear(EMB_DIM, bip_feat_dim)).to(device)
    model.bipartite_output = MaskLinear(bip_feat_dim, n_drugs, False).to(device)
    model.mask_H_transform = nn.Sequential(nn.Linear(bip_feat_dim, EMB_DIM)).to(device)
    
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR_RATE)
    
    print(f"\n{'='*50}\n  Training SafeDrug (seed={seed})\n{'='*50}")
    
    best_ja, best_state = 0, None
    
    for epoch in range(epochs):
        model.train()
        for step, input in enumerate(bundle['data_train']):
            loss = 0
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                
                def safe_seq(seq, mx):
                    return [max(0, min(int(v), mx-1)) for v in seq]
                
                for i, s in enumerate(seq_input):
                    seq_input[i] = [
                        safe_seq(s[0], bundle['voc_size'][0]),
                        safe_seq(s[1], bundle['voc_size'][1]),
                        safe_seq(s[2], bundle['voc_size'][2]),
                        s[3] if len(s) > 3 else [],
                        s[4] if len(s) > 4 else [],
                    ]
                
                loss_bce_target = np.zeros((1, bundle['voc_size'][2]))
                loss_bce_target[:, adm[2]] = 1
                
                result, loss_ddi = model(seq_input)
                loss_bce = F.binary_cross_entropy_with_logits(
                    result, torch.FloatTensor(loss_bce_target).to(device)
                )
                
                result_np = F.sigmoid(result).detach().cpu().numpy()[0]
                result_np[result_np >= 0.5] = 1
                result_np[result_np < 0.5] = 0
                y_label = np.where(result_np == 1)[0]
                current_ddi = calc_ddi_rate([[y_label]], bundle['ddi_adj'])
                
                if current_ddi <= TARGET_DDI:
                    loss = loss_bce
                else:
                    beta = min(0.0, 1 + (TARGET_DDI - current_ddi) / KP)
                    loss = beta * loss_bce + (1 - beta) * loss_ddi
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
            llprint(f"\rSafeDrug epoch {epoch+1}/{epochs}: {step+1}/{len(bundle['data_train'])}")
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} done")
        
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def train_llmdgtn(seed, bundle, epochs=EPOCH):
    """训练LLMDGTN模型"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = LLMDGTN(
        bundle['voc_size'], bundle['ehr_adj'], bundle['ddi_adj'],
        bundle['ddi_mask_H'], bundle['MPNNSet'], bundle['N_fingerprint'],
        bundle['average_projection'], emb_dim=EMB_DIM, device=device
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR_RATE)
    
    print(f"\n{'='*50}\n  Training LLMDGTN (seed={seed})\n{'='*50}")
    
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for step, input in enumerate(bundle['data_train']):
            for idx, adm in enumerate(input):
                seq_input = input[:idx+1]
                
                def safe_seq(seq, mx):
                    return [max(0, min(int(v), mx-1)) for v in seq]
                
                for i, s in enumerate(seq_input):
                    seq_input[i] = [
                        safe_seq(s[0], bundle['voc_size'][0]),
                        safe_seq(s[1], bundle['voc_size'][1]),
                        safe_seq(s[2], bundle['voc_size'][2]),
                        s[3] if len(s) > 3 else [],
                        s[4] if len(s) > 4 else [],
                    ]
                
                loss_bce_target = np.zeros((1, bundle['voc_size'][2]))
                loss_bce_target[:, adm[2]] = 1
                
                # 添加loss_multi_target（关键修复！）
                loss_multi_target = np.full((1, bundle['voc_size'][2]), -1)
                for item in adm[2]:
                    safe_item = max(0, min(int(item), bundle['voc_size'][2] - 1))
                    loss_multi_target[0][safe_item] = 1
                
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
                current_ddi = calc_ddi_rate([[y_label]], bundle['ddi_adj'])
                
                if current_ddi <= TARGET_DDI:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi  # 修复: 使用loss_multi
                else:
                    beta = min(0.0, 1 + (TARGET_DDI - current_ddi) / KP)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1-beta) * loss_ddi
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
            llprint(f"\rLLMDGTN epoch {epoch+1}/{epochs}: {step+1}/{len(bundle['data_train'])}")
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} done")
        
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  Multi-Model Visit Count Sensitivity Experiment")
    print("  Models: LR, GAMENet, SafeDrug, LLMDGTN")
    print("  Visit counts: 1-10")
    print("=" * 70)
    
    # 加载数据
    bundle = load_data()
    
    # 构建统计信息
    stats_path = '../data/mimic4_stats'
    count_conditional_prob_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_train'], stats_path)
    add_sparse_prior_guide_dp(bundle['data_eval'], stats_path)
    
    # 输出目录
    output_dir = os.path.join("saved", "LLMDGTN", "multi_model_visits")
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有结果: {model_name: {seed: {visit: metrics}}}
    all_results = {}
    
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n\n{'#'*70}")
        print(f"  ### RUN {run_idx+1}/{len(SEEDS)} | Seed={seed} ###")
        print(f"{'#'*70}")
        
        seed_results = {}
        
        # ====== 1. Train & Evaluate LR ======
        print(f"\n--- LR Model ---")
        lr_model = LRModel(bundle['voc_size'])
        lr_model.train(bundle['data_train'], bundle['voc_size'])
        lr_results = evaluate_by_exact_visits(lr_model, bundle['data_eval'], bundle['voc_size'],
                                               bundle['ddi_adj'], 'LR')
        seed_results['LR'] = lr_results
        del lr_model
        gc.collect()
        
        # ====== 2. Train & Evaluate GAMENet ======
        print(f"\n--- GAMENet ---")
        gamenet_model = train_gamenet(seed, bundle, epochs=EPOCH)
        gamenet_results = evaluate_by_exact_visits(gamenet_model, bundle['data_eval'],
                                                    bundle['voc_size'], bundle['ddi_adj'], 'GAMENet')
        seed_results['GAMENet'] = gamenet_results
        del gamenet_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ====== 3. Train & Evaluate SafeDrug ======
        print(f"\n--- SafeDrug ---")
        safedrug_model = train_safedrug(seed, bundle, epochs=EPOCH)
        safedrug_results = evaluate_by_exact_visits(safedrug_model, bundle['data_eval'],
                                                     bundle['voc_size'], bundle['ddi_adj'], 'SafeDrug')
        seed_results['SafeDrug'] = safedrug_results
        del safedrug_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # ====== 4. Train & Evaluate LLMDGTN ======
        print(f"\n--- LLMDGTN ---")
        llmdgtn_model = train_llmdgtn(seed, bundle, epochs=EPOCH)
        llmdgtn_results = evaluate_by_exact_visits(llmdgtn_model, bundle['data_eval'],
                                                    bundle['voc_size'], bundle['ddi_adj'], 'LLMDGTN')
        seed_results['LLMDGTN'] = llmdgtn_results
        del llmdgtn_model
        torch.cuda.empty_cache()
        gc.collect()
        
        all_results[f'seed_{seed}'] = seed_results
        
        # 打印当前run结果摘要
        print(f"\n\n=== RUN {run_idx+1} SUMMARY ===")
        print(f"{'Visits':<7}", end='')
        for m in MODELS_TO_TEST:
            print(f"| {m+' JA':>12}", end='')
        print()
        print('-' * 60)
        for v in VISIT_COUNTS:
            print(f"{v:<7}", end='')
            for m in MODELS_TO_TEST:
                ja = seed_results[m][v]['jaccard']
                pc = seed_results[m][v]['patient_cnt']
                print(f"| {ja:>10.4f}({pc:>3})", end='')
            print()
    
    # ================================================================
    # 计算多轮平均
    # ================================================================
    print(f"\n\n{'='*70}")
    print(f"  AVERAGED RESULTS over {len(SEEDS)} runs")
    print(f"{'='*70}")
    
    averaged = {}  # {model_name: {visit: metrics}}
    for model_name in MODELS_TO_TEST:
        averaged[model_name] = {}
        for v in VISIT_COUNTS:
            metrics = ['jaccard', 'prauc', 'avg_f1', 'ddi_rate', 'avg_med']
            averaged[model_name][v] = {}
            for metric in metrics:
                vals = [all_results[f'seed_{s}'][model_name][v][metric] for s in SEEDS]
                averaged[model_name][v][metric] = np.mean(vals)
            averaged[model_name][v]['patient_cnt'] = all_results[f'seed_{SEEDS[0]}'][model_name][v]['patient_cnt']
    
    # 打印表格
    print(f"\n{'='*120}")
    print(f"  JACCARD SCORE by Visits")
    print(f"{'='*120}")
    header = f"{'Visits':<7}"
    for m in MODELS_TO_TEST:
        header += f"| {m:>15}"
    header += "| Patients"
    print(header)
    print('-' * 95)
    for v in VISIT_COUNTS:
        row = f"{v:<7}"
        for m in MODELS_TO_TEST:
            row += f"| {averaged[m][v]['jaccard']:>15.4f}"
        row += f"| {int(averaged[MODELS_TO_TEST[0]][v]['patient_cnt']):>8}"
        print(row)
    
    # ================================================================
    # 保存结果
    # ================================================================
    import csv
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Visits', 'Patients', 'DDI_Rate', 'Jaccard', 'PRAUC', 'AVG_F1', 'AVG_Med'])
        for model_name in MODELS_TO_TEST:
            for v in VISIT_COUNTS:
                r = averaged[model_name][v]
                writer.writerow([
                    model_name, v, int(r['patient_cnt']),
                    f"{r['ddi_rate']:.5f}", f"{r['jaccard']:.5f}",
                    f"{r['prauc']:.5f}", f"{r['avg_f1']:.5f}", f"{r['avg_med']:.4f}",
                ])
    print(f"\nCSV saved: {csv_path}")
    
    # ================================================================
    # 绑图 - 4个子图 (Jaccard, F1, PRAUC, DDI)
    # ================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_config = [
        ('jaccard', 'Jaccard Score', axes[0, 0]),
        ('avg_f1', 'F1 Score', axes[0, 1]),
        ('prauc', 'PRAUC', axes[1, 0]),
        ('ddi_rate', 'DDI Rate', axes[1, 1]),
    ]
    
    x_vals = VISIT_COUNTS
    
    for metric_name, ylabel, ax in metrics_config:
        for model_name in MODELS_TO_TEST:
            color = MODEL_COLORS[model_name]
            marker = MODEL_MARKERS[model_name]
            y_vals = [averaged[model_name][v][metric_name] for v in x_vals]
            ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2,
                   markersize=7, label=model_name, alpha=0.85)
        
        ax.set_xlabel('Number of Visits', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'({chr(97+metrics_config.index((metric_name,ylabel,ax)))}) {ylabel} vs. Visits', fontsize=13)
        ax.set_xticks(x_vals)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Fig4.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved: {fig_path}")
    
    # 论文精简版图
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True)
    
    for idx, (metric_name, ylabel) in enumerate([('jaccard','Jaccard'),('avg_f1','F1 Score'),
                                                  ('prauc','PRAUC'),('ddi_rate','DDI Rate')]):
        ax = axes2[idx]
        for model_name in MODELS_TO_TEST:
            color = MODEL_COLORS[model_name]
            marker = MODEL_MARKERS[model_name]
            y_vals = [averaged[model_name][v][metric_name] for v in x_vals]
            ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=2.2,
                   markersize=7, label=model_name, alpha=0.9)
        
        ax.set_xlabel('Number of Visits', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'({chr(97+idx)}) {ylabel}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "Fig4_paper.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Paper figure saved: {fig2_path}")
    
    print("\n" + "=" * 70)
    print("  Experiment completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
