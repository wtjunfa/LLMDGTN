# -*- coding: utf-8 -*-
"""DNC 占位实现。

原 armr2 仓库的 DNC 仅在 DMNC 模型中使用，而 DMNC 不在本实验的基线清单
（LR/ECC/RETAIN/GAMENet/SafeDrug/COGNet/MICRON/DNMDR）中，因此此处提供一个
最小占位类，仅用于保证 `from dnc import DNC` 能正常导入。
"""
import torch.nn as nn


class DNC(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='gru', num_layers=1,
                 num_hidden_layers=1, nr_cells=16, cell_size=64, read_heads=1,
                 batch_first=True, gpu_id=0, independent_linears=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memories = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError("DNC 为占位实现，不应被调用（DMNC 不在实验清单中）")
