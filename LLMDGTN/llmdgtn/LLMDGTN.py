"""LLMDGTN 模型训练与评估主脚本。

LLMDGTN（LLM-enhanced Dynamic Graph Transformer Network）是一种用于安全药物推荐的
深度学习模型，融合了预训练大语言模型的语义知识与分子图神经网络的结构特征。

用法：
    python LLMDGTN.py --dim 64 --target_ddi 0.06 --cuda 0

数据准备与依赖安装请参考项目根目录下的 README.md。
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from torch.optim import Adam

from models import LLMDGTN
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN


def calc_ddi_rate(pred_records, ddi_mat):
    """计算预测药物组合的 DDI 率。"""
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

model_name = "LLMDGTN"
resume_path = ""  # Clear during training

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--ddi_lambda", type=float, default=2.0, help="DDI regularization weight (BCE stays dominant)")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=1208, help="random seed")
parser.add_argument("--wo_llm", action="store_true", default=False, help="ablation: remove LLM branch")
parser.add_argument("--wo_transformer", action="store_true", default=False, help="ablation: remove Transformer encoder")
parser.add_argument("--wo_mpnn", action="store_true", default=False, help="ablation: remove MPNN branch")
parser.add_argument("--wo_ddi_loss", action="store_true", default=False, help="ablation: remove DDI loss")
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed * 5 + 13)

# evaluate
def eval(model, data_eval, voc_size, epoch, ddi_adj_path, ddi_adj):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

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
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    ddi_rate = calc_ddi_rate(smm_record, ddi_adj)

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt
    )

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
    pickle.dump(pm_probs, open(output_path + '/pm_probs.empirical.p', 'wb'), -1)
    pickle.dump(pm_cond_probs, open(output_path + '/pm_cond_probs.empirical.p', 'wb'), -1)
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
                    indices_dpm.append((i, len(dx_ids) + len(proc_idx) + med))
                    prob = 0.0 if dm not in dm_cond_probs else dm_cond_probs[dm]
                    values_dpm.append(prob)

            for i, proc in enumerate(proc_idx):
                for j, med in enumerate(med_ids):
                    pm = str(proc) + ',' + str(med)
                    indices_dpm.append((len(dx_ids) + i, len(dx_ids) + len(proc_idx) + med))
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


def write_log(log_path, epoch, results):
    with open(log_path, 'a') as f:
        f.write(f"epoch {epoch + 1} --------------------------\n")
        f.write(f"DDI Rate: {results['ddi_rate']:.4f}, Jaccard: {results['ja']:.4f}, "
                f"PRAUC: {results['prauc']:.4f}, AVG_PRC: {results['avg_p']:.4f}, "
                f"AVG_RECALL: {results['avg_r']:.4f}, AVG_F1: {results['avg_f1']:.4f}, "
                f"AVG_MED: {results['avg_med']:.4f} "
                f"LOSS: {results['loss']:.4f}\n")
        f.write(f"best_Epoch: {results['best_epoch']}\n\n")


def main():
    # 数据目录（相对路径，从 src/ 目录运行）
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mimic-iv")
    data_path = os.path.join(data_dir, "records_final.pkl")
    voc_path = os.path.join(data_dir, "voc_final.pkl")
    ehr_adj_path = os.path.join(data_dir, "ehr_adj_final.pkl")
    ddi_adj_path = os.path.join(data_dir, "ddi_A_final.pkl")
    ddi_mask_path = os.path.join(data_dir, "ddi_mask_H.pkl")
    molecule_path = os.path.join(data_dir, "atc3toSMILES.pkl")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("Running device:", device)

    ehr_adj = np.array(dill.load(open(ehr_adj_path, "rb")))
    ddi_adj = np.array(dill.load(open(ddi_adj_path, "rb")))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))

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

    log_path = os.path.join("saved", args.model_name, "mimic4_log.txt")
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    stats_path = '../data/mimic4_stats'
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    count_conditional_prob_dp(data, stats_path, data_train)
    add_sparse_prior_guide_dp(data, stats_path, data_train)
    add_sparse_prior_guide_dp(data, stats_path, data_eval)
    add_sparse_prior_guide_dp(data, stats_path, data_test)


    voc_size = (len(voc['diag_voc'].idx2word), len(voc['pro_voc'].idx2word), len(voc['med_voc'].idx2word))
    print(f"Vocabulary size: diag={voc_size[0]}, pro={voc_size[1]}, med={voc_size[2]}")

    model = LLMDGTN(
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprint,
        average_projection,
        emb_dim=args.dim,
        device=device,
        use_llm=not args.wo_llm,
        use_transformer=not args.wo_transformer,
        use_mpnn=not args.wo_mpnn,
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    history["best_epoch"] = []
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print(f"\nepoch {epoch + 1} --------------------------")

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
                        step_adm[4] if len(step_adm) > 4 else []
                    ]

                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    safe_item = max(0, min(int(item), voc_size[2] - 1))
                    loss_multi_target[0][idx] = safe_item

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = calc_ddi_rate([[y_label]], ddi_adj)

                # 优化：BCE 始终主导，DDI 作为温和正则（替代原"切换式"loss，避免 DDI 超标时压制 PRAUC）
                loss_base = loss_bce + 0.05 * loss_multi
                if args.wo_ddi_loss:
                    loss = loss_base
                else:
                    # DDI 超标越多，惩罚越强；但始终不会压过 BCE，从而同时保持高 PRAUC 与低 DDI
                    ddi_slack = max(0.0, current_ddi_rate - args.target_ddi) / args.target_ddi
                    loss = loss_base + args.ddi_lambda * (1.0 + ddi_slack) * loss_ddi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, ddi_adj_path, ddi_adj)
        print(f"training time: {time.time() - tic:.2f}s, test time: {time.time() - tic2:.2f}s")

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)
        history["best_epoch"].append(best_epoch)

        if epoch >= 5:
            print(f"ddi: {np.mean(history['ddi_rate'][-5:]):.4f}, Med: {np.mean(history['med'][-5:]):.2f}, Ja: {np.mean(history['ja'][-5:]):.4f}, F1: {np.mean(history['avg_f1'][-5:]):.4f}, PRAUC: {np.mean(history['prauc'][-5:]):.4f}")

        results = {
            'ddi_rate': ddi_rate,
            'ja': np.mean(ja),
            'prauc': np.mean(prauc),
            'avg_p': np.mean(avg_p),
            'avg_r': np.mean(avg_r),
            'avg_f1': np.mean(avg_f1),
            'avg_med': avg_med,
            'best_epoch': best_epoch,
            'loss': loss.item()
        }
        write_log(log_path, epoch, results)

        torch.save(model.state_dict(),
                   os.path.join("saved", args.model_name, f"Epoch_{epoch}_TARGET_{args.target_ddi:.2}_JA_{ja:.4}_DDI_{ddi_rate:.4}.model"))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print(f"best_epoch: {best_epoch}")

    dill.dump(history, open(os.path.join("saved", args.model_name, "history.pkl"), "wb"))

    import pandas as pd
    history_df = pd.DataFrame(history)
    summary_df = pd.DataFrame({
        "DDI Rate": history_df["ddi_rate"],
        "Jaccard": history_df["ja"],
        "PRAUC": history_df["prauc"],
        "AVG_F1": history_df["avg_f1"],
        "AVG_MED": history_df["med"],
        "best_epoch": history_df["best_epoch"]
    })
    summary_df.to_csv(os.path.join("saved", args.model_name, "training_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n训练日志已保存：saved/{args.model_name}/training_summary.csv")

if __name__ == "__main__":
    main()