# config.py
# =================================================================
# 项目统一配置文件 (V2.2 - 冠军策略版)
# =================================================================

# ==== Toggle switches ====
# 是否启用自编码器(AE)生成“AI特征”
USE_AE: bool = True

# 是否训练 LGBM（关闭时仅构建/导出特征矩阵，便于单测AE或特征工程）
USE_LGBM: bool = True

# （可选）当关闭 LGBM 时，验证/持有集的占位预测值（用于跑通评估链）
NO_LGBM_BASELINE_PRED: float = 0.5  # 保持 0.5，AUC≈0.5 仅作占位

# --- 1. 文件与路径 ---
RAW_DATA_FILE = 'train_final_features.csv' # 用于模型开发和CV
HOLDOUT_DATA_FILE = 'test_final_features.csv' # 用于最终评估

RANKING_FILE = 'feature_ranking_v13.csv' 
LGBM_PARAMS_FILE = 'best_params_v13_lgbm.json'
AE_PARAMS_FILE = 'best_params_v13_ae.json'
OOF_OUTPUT_FILE = 'oof_predictions_v13_final.csv'

# --- 2. 验证策略 ---
N_SPLITS = 5
PURGE_SIZE = 1
EMBARGO_SIZE = 30
ANALYSIS_START_DATE_ID = 1055
# [核心升级] 定义在最终验证/预测时，只使用最后几折训练出的模型
N_LAST_FOLDS_TO_USE_INFERENCE = 3 
TUNE_LIMIT = 60   # 0 或缺省表示不限制

WARMUP_BLOCKS = 1
# --- 3. 特征工程与筛选 ---
MISSING_THRESHOLD = 0.3    # 从 0.30 改到 1.00：不再因缺失率而删列（交给掩码处理）
HIGH_MISS_MASK_THRES = 0.20  # >20% 缺失的列，一定生成 isna 掩码
USE_MISSING_MASK = True      # 打开缺失掩码特征
N_FEATURES_TO_SELECT_RFE = 100
N_TOP_FEATURES_TO_USE = 30 
AI_PREFIX = 'AE_'

# ===== Top-K 全局列池（基表）=====
TOPK_ENABLE: bool = True              # 开关：开启后裁基表列
TOPK_MODE = "global"   # 新增：折内TopK（无泄露）
TOPK_N = 80              # Top-K 数量（对基表生效）
TOPK_SOURCE: str = "file"         # "file" | "online"（先用文件）
# 可配置多个排名文件，按顺序尝试读取；若是 .txt 则每行 1 列名；若 .csv 优先找 "feature" 列
#TUNE_POOL_FILE = "dashboard_ae_lofo_top_features.csv,kf_out/feature_ranking_clean.csv"
TOPK_FILES: list[str] = [
    "dashboard_ae_lofo_top_features.csv", 
    "kf_out/feature_ranking_clean.csv"
]
TOPK_CORR_PRUNE_THR: float | None = 0.99   # |ρ|≥阈值视为共线（None 关闭）
TOPK_APPLY_SCOPE: str = "both"   # "ae_only" | "lgbm_base" | "both"

# 宽侧维度“护栏”（控制 AE_z 与天气的拼接规模；不影响 Top-K 基表）
WIDE_MAX_AE_Z: int = 100         # AE 编码最多取前多少维
WIDE_MAX_WEATHER_COLS: int = 8    # 天气派生最多保留多少列（按列顺序截断）

# --- 4. 目标定义 ---
# [核心修改] 将我们的主要训练目标，从硬标签切换到新的软标签
TARGET_COLUMNS = ['dls_target_1d', 'dls_target_3d', 'dls_target_5d']

# 我们仍然需要旧的action列，用于最终的AUC评估
ACTION_COLUMNS = ['action_1d', 'action_3d', 'action_5d']

# [核心升级] 定义与目标对应的原始收益列，用于计算样本权重
RESP_COLUMNS = ['resp_1d', 'resp_3d', 'resp_5d']
# 主目标也相应切换（虽然在回归任务中意义减弱，但保持一致性）
PRIMARY_TARGET_COLUMN = 'dls_target_1d' 

# --- 5. Optuna 配置 ---
OPTUNA_ENABLE: bool = False
N_TRIALS_LGBM = 50
N_TRIALS_AE = 10

# === AE Loss Weights ===
AE_LOSS_WEIGHT_CLS = 0.7    # 预测任务权重
AE_LOSS_WEIGHT_RECON = 0.3   # 重建任务权重

# ===== Kalman features integration (可控开关) =====
KF_ENABLE = False               # 是否启用KF特征并回
KF_DIR = "kf_out"                     # KF产出目录
KF_DEV_FILE = "dev_kf_features"       # 不带后缀；程序自动识别 .parquet/.csv
KF_HOLDOUT_FILE = "holdout_kf_features"


# KF merge safety (recommended)
KF_JOIN_KEYS = ["date_id", "row_id"]   # if unavailable, loader will fallback to ["row_id"] when possible
KF_STRICT_ALIGN = True                 # refuse merge if alignment looks wrong (prevents silent AUC collapse)
KF_ALLOW_INDEX_FALLBACK = True         # allow index-concat only when length matches + date spot-check passes
KF_MAX_ALLNAN_RATIO = 0.001            # too many all-NaN KF rows => treat as misalignment
KF_COL_PREFIX = "kf_"                  # used only if KF columns collide with base columns

# 是否在训练/验证阶段使用我们“去泄露+去冗余”的Top清单
USE_PRUNED_FEATURES_IN_TRAIN = True
PRUNED_FEATURE_LIST = "top_features_pruned.txt"

# Holdout是否也用这份清单（保守起见默认False，避免用dev排名“帮助”holdout）
USE_PRUNED_FEATURES_IN_HOLDOUT = True

# === Weather features ===
WEATHER_ENABLE = True
WEATHER_ENGINE = "hmm"          # "hmm" | "kmeans" | "gmm"
WEATHER_K = 4
WEATHER_OUTPUT = "post"         # "post" | "onehot" | "state"
WEATHER_STD_FLOOR = 1e-2
WEATHER_RANDOM_STATE = 42

# 选择“天气基底列”的正则：只从这些列里学习天气；避免把危险列/掩码列/EXT列作为基底
WEATHER_SPEC = [
    r".*(ema|sma|mom|rol_mean|rol_std|vol|skew|kurt|rank_|cs_).*"
]
WEATHER_EXCLUDE_REGEX = [
    r"^action_", r"^resp_", r"^dls_target_", r"^sample_weight",
    r"_isna$", r"^EXT_",
]

# === Tune 限制 (可选) ===
TUNE_LIMIT     = 60   # 仅在 “tune” 模式里，限制可见特征数量
WARMSTART_TOPK = 56

#时间衰减权重
TIME_DECAY_ENABLED = True
TIME_DECAY_HALF_LIFE_DAYS = 504   # 2 年半衰期（你已拍板）
TIME_DECAY_FLOOR = 0.2            # 最小权重地板
TIME_DECAY_NORMALIZE_PER_FOLD = True  # 每折归一化到均值=1
BASE_WEIGHT_FLOOR = 1e-3      # 或 1e-4
BASE_WEIGHT_CAP_PCT = 0.995   # 99.5 分位裁顶；可先设 None 关闭

#preprocess部分
PREPROCESS_IMPUTE_STRATEGY = "median"   # or "mean"
PREPROCESS_CENTER = True
PREPROCESS_SCALE = True
PREPROCESS_STD_FLOOR = 0.01
PREPROCESS_CLIP_Z = 6.0                 # None 关闭裁剪
PREPROCESS_DROP_NAN_COL_RATE = 0.30     # 训练折内缺失率 > 30% 的列将被丢弃


# === Decision Layer (offline proxy & future submission defaults) ===
DECISION_MODE = "tilt"
DECISION_NONLINEAR = "tanh"
DECISION_COEF = 0.019790743235863214
DECISION_DECAY = 0
DECISION_THRESHOLD = 0.06115587211018596         # 死区：|p-0.5| 小于该值则不交易
DECISION_VOL_WINDOW = 13          # 波动率滚动窗口长度（天）

DECISION_CLIP_LOW = 0.0
DECISION_CLIP_HIGH = 2.0          # 题目规定的 0~2 仓位范围

DECISION_COVERAGE_FLOOR = 0.05    # 若交易覆盖率低于这个值，在 proxy 上给轻微惩罚提示

# === Regime-aware overlay（Block3 新增，用天气/HMM 调整仓位）===
# 注意：这些只影响 offline Sharpe proxy，不会改变 AUC 训练流程
DECISION_USE_WEATHER_REGIME = True   # 是否启用 Regime 缩放
DECISION_REGIME_BULL_SCALE = 1.2     # “相对多头”状态倍率
DECISION_REGIME_NEUTRAL_SCALE = 1.0  # 中性状态倍率
DECISION_REGIME_BEAR_SCALE = 0.7     # “相对空头/回避”状态倍率

# 用 states 的平均收益来排序，按分位数切三档（底档=熊，中档=中性，上档=牛）
DECISION_REGIME_QUANTILES = (0.33, 0.67)
DECISION_REGIME_MIN_STATE_SAMPLES = 200  # 某个 state 样本数太少就直接放弃 Regime overlay

#LGBM
LGBM_USE_WEATHER = True   # 默认 True；设 False 就回到以前的行为

# === [AUTO DECISION PARAMS START] ===
# auto-generated at: 2025-12-15T23:55:25
DECISION_CLIP_HIGH = 2.0
DECISION_CLIP_LOW = 0.0
DECISION_COEF = 2.2031716082050545
DECISION_DECAY = 0.0
DECISION_THRESHOLD = 0.0014998037313254282
DECISION_USE_WEATHER_REGIME = False
MAX_INVESTMENT = 2.0
MIN_INVESTMENT = 0.0
# === [AUTO DECISION PARAMS END] ===
# ==============================
# Bundle export (Kaggle inference)
# ==============================
# Set EXPORT_BUNDLE_DIR to a folder name to enable exporting per-fold models during validate/train.
# Example: EXPORT_BUNDLE_DIR = "hull_model_bundle"
EXPORT_BUNDLE_DIR = None
EXPORT_BUNDLE_FOLDS = "last_k"       # "last_k" or "all"
EXPORT_BUNDLE_COPY_CODE = True       # copy preprocess.py/weather.py/encoder.py/config.py into the bundle
EXPORT_BUNDLE_OVERWRITE = True
