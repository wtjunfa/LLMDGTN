# -*- coding: utf-8 -*-
"""ECC (Ensemble of Classifier Chains) 基线：以诊断+手术多热向量为输入的多标签分类"""
import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from collections import defaultdict
import os
import argparse
from util import Metrics
import time

import sys
sys.path.append('..')
from util import multi_label_metric

model_name = 'ECC'

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--datadir', type=str, default="../data/", help='dimension')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--seed', type=int, default=1029, help='seed')

args = parser.parse_args()
args.MIMIC = 3
if not os.path.exists(os.path.join("saved", args.model_name)):
    os.makedirs(os.path.join("saved", args.model_name))


def create_dataset(data, diag_voc, pro_voc, med_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(pro_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]
            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1
            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1
            X.append(multi_hot_input)
            y.append(multi_hot_output)
    return np.array(X), np.array(y)


data_path = os.path.join(args.datadir, 'records_final.pkl')
voc_path = os.path.join(args.datadir, 'voc_final.pkl')

data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
metric_obj = Metrics(data, med_voc, args)

for epoch in range(1):
    np.random.seed(args.seed)
    np.random.shuffle(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point + eval_len:]
    data_test = data[split_point:split_point + eval_len]

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    model = LogisticRegression(max_iter=1000)

    # 过滤训练集中只有一个类别的标签列（罕见药物全 0），避免 ClassifierChain 报错
    col_sum = train_y.sum(axis=0)
    valid_cols = np.where((col_sum > 0) & (col_sum < len(train_y)))[0]
    output_len = train_y.shape[1]

    classifier = ClassifierChain(model)

    tic = time.time()
    classifier.fit(train_X, train_y[:, valid_cols])
    fittime = time.time() - tic

    test_sample = test_X
    y_sample = test_y
    y_pred_valid = classifier.predict(test_sample)
    pretime = time.time() - tic

    # predict_proba 返回 (n_samples, n_valid_labels)
    y_prob_valid = classifier.predict_proba(test_sample)

    # 重建完整维度（被过滤的单类别列预测为 0）
    y_pred = np.zeros((len(test_sample), output_len))
    y_pred[:, valid_cols] = y_pred_valid
    y_prob = np.zeros((len(test_sample), output_len))
    y_prob[:, valid_cols] = y_prob_valid

    metric_obj.set_data(y_sample, y_pred, y_prob, save=args.Test)
    ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run()

    ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')
    ddi_A = np.array(dill.load(open(ddi_adj_path, 'rb')))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt

    print('Epoch: {}, DDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        epoch, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, med_cnt / visit_cnt))

    history = defaultdict(list)
    history['fittime'].append(fittime)
    history['pretime'].append(pretime)
    history['jaccard'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)
    history['med'].append(med_cnt / visit_cnt)

dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
print('ECC done. DDI: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, F1: {:.4f}'.format(
    ddi_rate, ja, prauc, avg_f1))
