"""
create_manual_ris.py
Manually writes RIS files for the 11 papers that could not be
downloaded automatically (IEEE CrossRef gaps + arXiv timeouts).
All metadata is sourced from the subagent search results in References.md.
"""

import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MANUAL = {

# ── IEEE papers (CrossRef 404) ───────────────────────────────────────────────

"ref09_jin2022.ris": """\
TY  - JOUR
AU  - Jin, Rui
AU  - Chen, Zhen
AU  - Qi, Yue
AU  - Yin, Zheng
TI  - Bi-LSTM-Based Two-Stream Network for Machine Remaining Useful Life Prediction
JO  - IEEE Transactions on Instrumentation and Measurement
VL  - 71
PY  - 2022
DO  - 10.1109/TIM.2022.3163761
UR  - https://doi.org/10.1109/TIM.2022.3163761
N1  - [ref09] RMSE FD001~12.38 FD002~19.57 FD003~12.83 FD004~19.98
ER  - """,

"ref12_li2023.ris": """\
TY  - JOUR
AU  - Li, Jian
AU  - Li, Xiaoli
AU  - He, David
TI  - Remaining Useful Life Prediction of Turbofan Engines Using CNN-LSTM-SAM Approach
JO  - IEEE Sensors Journal
VL  - 23
IS  - 10
PY  - 2023
DO  - 10.1109/JSEN.2023.3243540
UR  - https://doi.org/10.1109/JSEN.2023.3243540
N1  - [ref12] CNN+LSTM+self-attention for FD002/FD004 multi-operating-point
ER  - """,

"ref13_wang2023.ris": """\
TY  - JOUR
AU  - Wang, Han
AU  - Liu, Zhen
AU  - Peng, Daiyue
AU  - Cheng, Zeyi
TI  - Comprehensive Dynamic Structure Graph Neural Network for Aero-Engine Remaining Useful Life Prediction
JO  - IEEE Transactions on Instrumentation and Measurement
VL  - 72
PY  - 2023
DO  - 10.1109/TIM.2023.3312337
UR  - https://doi.org/10.1109/TIM.2023.3312337
N1  - [ref13] Dynamic inter-sensor graph; validated on CMAPSS and N-CMAPSS
ER  - """,

"ref14_you2024.ris": """\
TY  - JOUR
AU  - You, Kang
AU  - Lu, Shuai
AU  - Zhao, Xinghao
AU  - Chen, Ran
TI  - A 3-D Attention-Enhanced Hybrid Neural Network for Turbofan Engine Remaining Life Prediction Using CNN and BiLSTM Models
JO  - IEEE Sensors Journal
VL  - 24
IS  - 3
PY  - 2024
DO  - 10.1109/JSEN.2023.3335994
UR  - https://doi.org/10.1109/JSEN.2023.3335994
N1  - [ref14] 3-D spatial-channel attention; recent CMAPSS SOTA with handcrafted degradation auxiliary inputs
ER  - """,

# ── arXiv papers ─────────────────────────────────────────────────────────────

"ref22_abdullah2026.ris": """\
TY  - UNPB
AU  - Abdullah, Mohammedali E. B.
TI  - Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps
PY  - 2026
AN  - arXiv:2604.13459
UR  - https://arxiv.org/abs/2604.13459
N1  - [ref22] Uses NASA asymmetric loss (H7-L2) on CMAPSS FD001; RMSE 17.52, clip=130
ER  - """,

"ref23_kim2022_revin.ris": """\
TY  - CONF
AU  - Kim, Taesung
AU  - Kim, Jinhee
AU  - Tae, Yunwon
AU  - Park, Cheonbok
AU  - Choi, Jang-Ho
AU  - Choo, Jaegul
TI  - Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
T2  - International Conference on Learning Representations (ICLR 2022)
PY  - 2022
AN  - arXiv:2110.02454
UR  - https://openreview.net/forum?id=cGDAkQo1C0p
N1  - [ref23] Original RevIN paper; proposes symmetric per-instance norm/denorm — evaluated as N7 in H5
ER  - """,

"ref25_berthelier2026.ris": """\
TY  - UNPB
AU  - Berthelier, Gaspard
TI  - On the Role of Reversible Instance Normalization
PY  - 2026
UR  - https://consensus.app/papers/details/f2feae60790d5d18a0c5b5358099fbd4/
N1  - [ref25] Ablation revealing RevIN components redundant when normalization challenges are temporal/spatial/conditional; supports H5 finding that RevIN fails on multi-condition FD002/FD004. Author/arXiv ID not confirmed — verify before submission.
ER  - """,

"ref26_noise_signal2025.ris": """\
TY  - UNPB
TI  - Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting
PY  - 2025
AN  - arXiv:2510.04667
UR  - https://arxiv.org/abs/2510.04667
N1  - [ref26] Identifies RevIN failure modes when instance statistics encode non-degradation variance (regime shifts); mechanism directly applies to H5 FD002/FD004 multi-condition failure. Author list not retrieved — verify before submission.
ER  - """,

"ref27_early_fault2026.ris": """\
TY  - UNPB
TI  - Early Fault Detection on CMAPSS with Unsupervised LSTM Autoencoders
PY  - 2026
AN  - arXiv:2601.10269
UR  - https://arxiv.org/abs/2601.10269
N1  - [ref27] Regression-based operating-condition normalization for turbofan; shows raw normalization inadequate for FD002/FD004 (supports H5). Also motivates unsupervised fault onset detection for H6. Author list not retrieved — verify before submission.
ER  - """,

"ref34_bayesian_np2026.ris": """\
TY  - UNPB
TI  - Prognostics of Multisensor Systems with Unknown and Unlabeled Failure Modes via Bayesian Nonparametric Process Mixtures
PY  - 2026
AN  - arXiv:2602.19263
UR  - https://arxiv.org/abs/2602.19263
N1  - [ref34] Dirichlet process mixture model for label-free failure mode discovery; contextualises H6 GMM-based unsupervised fault separation. Author list not retrieved — verify before submission.
ER  - """,

"ref41_chung2021.ris": """\
TY  - CONF
AU  - Chung, Youngseog
AU  - Neiswanger, Willie
AU  - Char, Ian
AU  - Schneider, Jeff
TI  - Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification
T2  - Advances in Neural Information Processing Systems (NeurIPS 2021)
PY  - 2021
AN  - arXiv:2010.09964
UR  - https://arxiv.org/abs/2010.09964
N1  - [ref41] Foundational critique of pinball loss; shows blind minimisation does not guarantee calibrated intervals — supports H7 L6 (Pinball) finding of no significant gain
ER  - """,

}

# ── write files ───────────────────────────────────────────────────────────────
created, skipped = [], []
for filename, content in MANUAL.items():
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        skipped.append(filename)
        continue
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    created.append(filename)
    print(f"  [written] {filename}")

print(f"\nCreated: {len(created)}, Skipped (already OK): {len(skipped)}")
