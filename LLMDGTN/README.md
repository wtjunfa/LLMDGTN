# LLMDGTN 重新跑数据 + 重新绘图

本文件夹用于**重新跑一遍药物推荐实验**，并按新的图例要求重新生成论文三张图（对比实验 / 消融实验 / 就诊次数）。

## 目录结构

```
LLMDGTN_重新跑数据/
├── baselines/            # 基线训练代码（LR/ECC/Retain/GAMENet/SafeDrug/COGNet/MICRON）
│   └── baselines/        # 各基线脚本 + 依赖（util.py 等）
├── llmdgtn/              # LLMDGTN 模型代码（LLMDGTN.py / models.py 等，已加消融开关）
├── data/mimic-iv/        # 数据（已转好 pkl：records_final.pkl 等）
├── plots/                # 画图脚本（图1/图2/图3，图例已按新要求设好）
├── logs/                 # 各模型日志 CSV（画图数据源）
├── results/              # 生成的图
├── collect_logs.py       # 收集日志脚本
└── README.md
```

## 环境

- **Python**：`D:\Anaconda\python.exe`（含 torch / numpy / pandas / matplotlib / rdkit / dill / scikit-learn）
- GPU 可选（`--cuda 0`），无 GPU 自动回退 CPU

## 数据（已准备好）

`data/mimic-iv/` 下已包含全部所需 pkl：
`records_final.pkl`、`voc_final.pkl`、`ehr_adj_final.pkl`、`ddi_A_final.pkl`、`ddi_mask_H.pkl`、`atc3toSMILES.pkl`

数据划分即 **4:1:1**（训练 2/3，验证/测试各 1/6）。

---

## 第一步：跑基线（8 个模型）

在 `baselines/baselines/` 目录下运行（每个模型一个命令）：

```bash
cd baselines/baselines

python LR.py        --datadir ../../data/mimic-iv
python ECC.py       --datadir ../../data/mimic-iv
python Retain.py    --datadir ../../data/mimic-iv
python GAMENet.py   --datadir ../../data/mimic-iv
python SafeDrug.py  --datadir ../../data/mimic-iv
python COGNet.py    --datadir ../../data/mimic-iv
python MICRON.py    --datadir ../../data/mimic-iv
```

DNMDR 单独在 `baselines/dnmdr/` 目录运行：

```bash
cd baselines/dnmdr
python DNMDR.py --dim 64 --target_ddi 0.06 --cuda 0
```

> 共 8 个基线已就位（LR / ECC / Retain / GAMENet / SafeDrug / COGNet / MICRON / DNMDR）。**AKA-SafeMed 无公开代码**（见文末"待补"）。

## 第二步：跑 LLMDGTN（完整模型）

```bash
cd llmdgtn
python LLMDGTN.py --dim 64 --target_ddi 0.06 --cuda 0
```

## 第三步：跑消融（5 个变体）

```bash
cd llmdgtn
python LLMDGTN.py --dim 64 --wo_llm            --model_name LLMDGTN_woLLM
python LLMDGTN.py --dim 64 --wo_transformer    --model_name LLMDGTN_woTransformer
python LLMDGTN.py --dim 64 --wo_mpnn           --model_name LLMDGTN_woMPNN
python LLMDGTN.py --dim 64 --wo_ddi_loss       --model_name LLMDGTN_woDDILoss
python LLMDGTN.py --dim 64 --wo_llm --wo_transformer --model_name LLMDGTN_woLLM_Transformer
```

> 每个变体**必须**指定不同的 `--model_name`（文件名已与画图脚本 `plot_fig2_ablation.py` 的 `ABLATION_ORDER` 对应），否则会覆盖彼此的 saved 结果。

## 第四步：收集日志

跑完所有模型后，回到根目录：

```bash
cd ..
python collect_logs.py
```

它会自动把各模型的训练日志统一转成 `logs/*.csv`。

## 第五步：画图

```bash
python plots/plot_fig1_comparison.py   # 图1：对比实验
python plots/plot_fig2_ablation.py     # 图2：消融实验
python plots/plot_fig3_visits.py       # 图3：就诊次数
```

生成的图在 `results/` 目录。

---

## 图例说明（已按要求配置）

- **图1**：LR, ECC, DNMDR, GAMENet, COGNet, MICRON, SafeDrug, RETAIN, AKA-SM, LLMDGTN
- **图2**：LT w/o LLM, LT w/o TR, LT w/o MPNN, LT w/o DDI Loss, LT w/o LLM_TR, LT（LT=LLMDGTN，TR=Transformer，LLM_TR=LLM+Transformer）
- **图3**：LR, GAMENet, SafeDrug, LLMDGTN

图例在 `plots/*.py` 顶部的 `MODEL_ORDER` / `ABLATION_ORDER` 变量里，可随时改。

## 沿用旧值（无需重跑）的部分

以下两类日志因**无法重跑**，沿用 `logs/` 里的旧数据：

| 日志 | 用途 | 原因 |
|------|------|------|
| `AKA-SafeMed.csv` | 图1 的 AKA-SM 曲线 | 无公开代码 |
| `results.csv` | 图3 就诊次数对比 | 实验脚本依赖缺失（`models-checkpoint.py`） |

`collect_logs.py` 已加入保护：每次收集日志时自动备份到 `data/*_old.csv`；若 `logs/` 被误清空会自动恢复。

其余 9 个模型（8 基线 + LLMDGTN）均已可重跑。
