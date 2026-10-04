"""LLMDGTN 模型定义：包含 GCN、GAT、MaskLinear、分子图神经网络（MPNN）以及
LLMDGTN 主模型等模块。"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import egcn_o
import utils as u
from layers import GAT, GraphConvolution


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self, x, adj):
        node_embedding = self.gcn1(x, adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, adj)
        return node_embedding

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GraphAT(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        super(GraphAT, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GAT(voc_size, emb_dim, emb_dim, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GAT(emb_dim, emb_dim, emb_dim, 1)

    def forward(self, x):
        node_embedding = self.gcn1(x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i + m, j:j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = hs
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return molecular_vectors


class LLMDGTN(nn.Module):
    def __init__(
            self,
            vocab_size,
            ehr_adj,
            ddi_adj,
            ddi_mask_H,
            MPNNSet,
            N_fingerprints,
            average_projection,
            emb_dim=64,
            device=torch.device("cuda:0"),
            use_llm=True,
            use_transformer=True,
            use_mpnn=True,
    ):
        super(LLMDGTN, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.emb_dim = emb_dim
        self.use_llm = use_llm
        self.use_transformer = use_transformer
        self.use_mpnn = use_mpnn

        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(3)])
        for embedding_layer in self.embeddings:
            nn.init.xavier_uniform_(embedding_layer.weight)

        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(2)])
        self.m_encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True)])

        # query 输入 = 诊断 GRU(2*emb_dim) + 手术 GRU(2*emb_dim) + 药物历史 GRU(emb_dim)
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(5 * emb_dim, emb_dim))

        med_num = vocab_size[2]
        self.gcn = GCN(voc_size=med_num, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ehr_gcn = GraphAT(voc_size=med_num, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GraphAT(voc_size=med_num, emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1)).to(device)
        nn.init.constant_(self.inter, 0.5)

        self.w5 = nn.Parameter(torch.FloatTensor(1)).to(device)
        nn.init.constant_(self.w5, 0.5)

        # 可学习 LLM 融合权重（初始 0.5，与原固定权重一致）
        self.llm_fuse_weight = nn.Parameter(torch.tensor(0.5, device=device))

        self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], med_num, False)
        self.mask_H_transform = nn.Sequential(nn.Linear(ddi_mask_H.shape[1], emb_dim))

        if use_mpnn:
            self.MPNN_molecule_Set = list(zip(*MPNNSet))
            self.MPNN_emb = MolecularGraphNeuralNetwork(
                N_fingerprints, emb_dim, layer_hidden=2, device=device
            ).forward(self.MPNN_molecule_Set)
            self.MPNN_emb = torch.mm(
                average_projection.to(device=self.device),
                self.MPNN_emb.to(device=self.device),
            )
            self.MPNN_emb.to(device=self.device)
        else:
            # wo_mpnn：用可学习嵌入表替代分子图结构编码
            self.drug_embedding = nn.Embedding(med_num, emb_dim).to(device)
            nn.init.xavier_uniform_(self.drug_embedding.weight)

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 2, med_num))

        self.d_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4)
        self.p_multi_head_attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4)
        self.key_output = nn.Linear(med_num, med_num)
        self.key_layernorm = nn.LayerNorm(med_num)

        # 预训练 LLM 药物嵌入（由 scripts/generate_llm_embedding.py 生成，见 README.md）
        if use_llm:
            llm_emb_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data", "output", "drug_llama_embedding.pt"
            )
            self.drug_llm_emb = torch.load(llm_emb_path, map_location=device)[:med_num]
            self.drug_llm_emb = F.normalize(self.drug_llm_emb, p=2, dim=1)
            llm_dim = self.drug_llm_emb.shape[1]
            self.llm_proj = nn.Linear(llm_dim, emb_dim).to(device)

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=4, dim_feedforward=128,
                dropout=0.1, activation="relu", batch_first=True
            ).to(device)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)

        self.init_weights()

    def forward(self, input):
        med_num = self.vocab_size[2]
        voc_med = list(range(med_num))
        o1, o2 = None, None

        safe_input = []
        for adm in input:
            d_safe = [max(0, min(int(x), self.vocab_size[0] - 1)) for x in adm[0]]
            p_safe = [max(0, min(int(x), self.vocab_size[1] - 1)) for x in adm[1]]
            m_safe = [max(0, min(int(x), med_num - 1)) for x in adm[2]]
            safe_adm = [d_safe, p_safe, m_safe]
            if len(adm) > 3:
                safe_adm.append(adm[3])
            if len(adm) > 4:
                safe_adm.append(adm[4])
            safe_input.append(safe_adm)
        input = safe_input

        i1_seq, i2_seq, i3_seq = [], [], []

        def sum_embedding(emb):
            return emb.sum(1).unsqueeze(0)

        for adm in input:
            i1 = sum_embedding(
                self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))))
            i2 = sum_embedding(
                self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))))
            # 药物历史编码：历史用药信息有助于下一次用药推荐
            if len(adm[2]) > 0:
                i3 = sum_embedding(
                    self.dropout(self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(0).to(self.device))))
            else:
                i3 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)
        i1_seq = torch.cat(i1_seq, dim=1)
        i2_seq = torch.cat(i2_seq, dim=1)
        i3_seq = torch.cat(i3_seq, dim=1)
        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        o3, _ = self.m_encoders[0](i3_seq)
        patient_representations = torch.cat([o1, o2, o3], dim=-1).squeeze(0)

        queries = self.query(patient_representations)
        query = queries[-1:]

        if self.use_mpnn:
            drug_memory = self.MPNN_emb[:med_num]
        else:
            drug_memory = self.drug_embedding(torch.arange(med_num).to(self.device))
        drug_memory = torch.nan_to_num(drug_memory)

        # 修复：去掉 LLM 和 Transformer 时，MPNN 嵌入因分子图对原子向量"求和"尺度爆炸（可达数百），
        # 无 LayerNorm/normalize 压制会导致 forward 数值爆炸、模型冻结。此处做 L2 归一化（与 LLM 嵌入一致）。
        if not self.use_llm and not self.use_transformer:
            drug_memory = F.normalize(drug_memory, p=2, dim=-1)

        combined = drug_memory
        if self.use_llm:
            llm_emb_proj = self.llm_proj(self.drug_llm_emb)
            llm_emb_proj = torch.nan_to_num(llm_emb_proj)
            llm_attn_weight = torch.sigmoid(torch.mm(query, llm_emb_proj.t()))
            llm_fact = torch.mm(llm_attn_weight, llm_emb_proj)
            llm_fact = llm_fact.expand(drug_memory.shape[0], -1)
            combined = drug_memory + self.llm_fuse_weight * llm_fact

        if self.use_transformer:
            combined = combined.unsqueeze(0)
            drug_memory = self.transformer_encoder(combined).squeeze(0)
            drug_memory = torch.nan_to_num(drug_memory)
        else:
            drug_memory = combined

        key_weights = torch.sigmoid(torch.mm(query, drug_memory.t()))
        fact1 = torch.mm(key_weights, drug_memory)
        fact1 = torch.nan_to_num(fact1)

        result = self.output(torch.cat([query, fact1], dim=-1))
        result = torch.nan_to_num(result)

        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj[:med_num, :med_num]).sum()

        return result, batch_neg

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'llm_proj'):
            nn.init.xavier_uniform_(self.llm_proj.weight)