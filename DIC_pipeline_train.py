import argparse
import json
import logging
import os
import random
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import timm


LOGGER = logging.getLogger(__name__)

# ----------------- Global config -----------------
CANONICAL_CLASSES = ["normal", "lsil", "hsil", "cancer"]

# Dataset-specific clinical feature columns.
# If your public dataset uses different column names, modify these constants.
CLINICAL_FEATURE_COLS = [
    "醋酸白上皮",
    "镶嵌",
    "点状血管",
    "边界",
    "醋白上皮内部边界征",
    "不典型血管",
]

# Explicit CNN features used by the ML fusion model.
CNN_BASIC_FEATURE_COLS = [
    "cnn_p_normal",
    "cnn_p_lsil",
    "cnn_p_hsil",
    "cnn_p_cancer",
    "cnn_pred_class",
    "cnn_severity",
]


# ----------------- Utility -----------------
def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fusion pipeline: CNN probabilities + CNN embedding PCA + clinical signs "
            "for 4-class colposcopic diagnosis."
        )
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to the input Excel file. Must contain split/class_name/image_path and clinical feature columns.",
    )
    parser.add_argument(
        "--cnn_ckpt",
        type=str,
        required=True,
        help="Path to the trained CNN checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ml_results_pca",
        help="Directory to save metrics, plots, feature importance, predictions, and model bundle.",
    )
    parser.add_argument(
        "--predictions_excel_path",
        type=str,
        default=None,
        help=(
            "Path to save Excel with appended predictions. "
            "If omitted, save to <output_dir>/predictions_with_ml.xlsx instead of overwriting the input Excel."
        ),
    )
    parser.add_argument(
        "--overwrite_input_excel",
        action="store_true",
        help="Overwrite the original Excel file. Disabled by default for safer public usage.",
    )
    parser.add_argument(
        "--ml_model_path",
        type=str,
        default=None,
        help="Optional explicit path to save ML bundle (.pkl). Default: <output_dir>/ml_fusion_bundle.pkl",
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        default="lightgbm",
        choices=["auto", "lightgbm", "xgboost", "random_forest", "svm"],
        help="Which ML model to use for fusion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference/training device selection.",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap_roc_grid_points",
        type=int,
        default=201,
        help="Number of fixed FPR grid points used to compute mean ROC.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


class CnnInferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            LOGGER.warning("Failed to read image %s: %s", img_path, exc)
            image = Image.new("RGB", (384, 384))
        if self.transform:
            image = self.transform(image)
        return image, idx


def build_cnn_model(num_classes: int, drop_rate: float) -> nn.Module:
    LOGGER.info("Building backbone: tf_efficientnetv2_b0")
    model = timm.create_model(
        "tf_efficientnetv2_b0",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn_checkpoint(model: nn.Module, ckpt_path: Path) -> Dict[int, str]:
    LOGGER.info("Loading CNN checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)

    label_mapping = ckpt.get("label_mapping", None) if isinstance(ckpt, dict) else None
    if label_mapping is None:
        return {i: c for i, c in enumerate(CANONICAL_CLASSES)}

    idx_to_label: Dict[int, str] = {}
    for label_name, label_idx in label_mapping.items():
        idx_to_label[int(label_idx)] = str(label_name).strip().lower()
    return idx_to_label


def run_cnn_inference_and_extract(
    df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    input_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = CnnInferDataset(df, tfm)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        persistent_workers=(num_workers > 0),
    )

    model.eval()

    embedding_dim = model.classifier.in_features
    all_embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)
    all_probs = np.zeros((len(dataset), 4), dtype=np.float32)

    with torch.no_grad():
        pbar = tqdm(loader, desc="CNN Feature Extraction", ncols=100)
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)

            features = model.forward_features(images)
            embeddings = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features, pre_logits=False)
            probs = torch.softmax(logits, dim=1)

            emb_np = embeddings.detach().cpu().numpy()
            prob_np = probs.detach().cpu().numpy()

            for i, row_idx in enumerate(idxs.numpy()):
                all_embeddings[row_idx] = emb_np[i]
                all_probs[row_idx] = prob_np[i]

    return df, all_embeddings, all_probs


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def encode_labels(series: pd.Series) -> pd.Series:
    mapping = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}
    encoded = series.map(lambda x: mapping.get(normalize_label(x), None))
    if encoded.isnull().any():
        bad = series[encoded.isnull()].unique()
        raise ValueError(f"Found unknown class_name values: {bad}")
    return encoded.astype(int)


def prepare_features_with_pca(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cnn_probs: np.ndarray,
    n_components: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], PCA, pd.DataFrame, pd.DataFrame]:
    LOGGER.info(
        "Preparing fusion features. Embedding shape=%s, probs shape=%s",
        embeddings.shape,
        cnn_probs.shape,
    )

    if np.isnan(embeddings).any():
        LOGGER.warning("NaNs detected in embeddings. Replacing with 0.")
        embeddings = np.nan_to_num(embeddings)

    df = df.copy()
    df["cnn_p_normal"] = cnn_probs[:, 0]
    df["cnn_p_lsil"] = cnn_probs[:, 1]
    df["cnn_p_hsil"] = cnn_probs[:, 2]
    df["cnn_p_cancer"] = cnn_probs[:, 3]

    pred_idx = cnn_probs.argmax(axis=1)
    df["cnn_pred_class"] = pred_idx
    severity_weights = np.arange(4, dtype=np.float32)
    df["cnn_severity"] = (cnn_probs * severity_weights.reshape(1, -1)).sum(axis=1)

    split_series = df["split"].astype(str).str.strip().str.lower()
    train_mask = (split_series == "train").values
    val_mask = split_series.isin(["val", "valid", "validation", "test"]).values

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train or validation split is empty.")

    X_train_emb = embeddings[train_mask]
    X_val_emb = embeddings[val_mask]

    max_valid_components = min(X_train_emb.shape[0], X_train_emb.shape[1])
    if n_components > max_valid_components:
        LOGGER.warning(
            "Requested pca_components=%d exceeds valid maximum=%d. Using %d instead.",
            n_components,
            max_valid_components,
            max_valid_components,
        )
        n_components = max_valid_components

    LOGGER.info("Fitting PCA on train embeddings. n_components=%d", n_components)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_emb)
    X_val_pca = pca.transform(X_val_emb)
    LOGGER.info("PCA explained variance ratio sum = %.4f", pca.explained_variance_ratio_.sum())

    pca_cols = [f"vis_feat_{i}" for i in range(n_components)]
    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_cols, index=df.index[train_mask])
    df_val_pca = pd.DataFrame(X_val_pca, columns=pca_cols, index=df.index[val_mask])

    clinical_train = df.loc[train_mask, CLINICAL_FEATURE_COLS].fillna(0)
    clinical_val = df.loc[val_mask, CLINICAL_FEATURE_COLS].fillna(0)

    cnn_train = df.loc[train_mask, CNN_BASIC_FEATURE_COLS]
    cnn_val = df.loc[val_mask, CNN_BASIC_FEATURE_COLS]

    X_train_final = pd.concat([clinical_train, cnn_train, df_train_pca], axis=1)
    X_val_final = pd.concat([clinical_val, cnn_val, df_val_pca], axis=1)

    y_train = encode_labels(df.loc[train_mask, "class_name"])
    y_val = encode_labels(df.loc[val_mask, "class_name"])

    feature_names = list(X_train_final.columns)
    df_train = df.loc[train_mask].copy()
    df_val = df.loc[val_mask].copy()

    return X_train_final, X_val_final, y_train, y_val, feature_names, pca, df_train, df_val


def train_ml_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int, choice: str):
    choice = choice.lower()

    def make_lightgbm():
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install it or choose another --ml_model."
            ) from exc

        return LGBMClassifier(
            objective="multiclass",
            num_class=4,
            class_weight="balanced",
            random_state=seed,
            n_estimators=600,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=31,
            n_jobs=-1,
            verbose=-1,
        ), "LightGBM"

    def make_xgboost():
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it or choose another --ml_model."
            ) from exc

        return XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            random_state=seed,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        ), "XGBoost"

    def make_svm():
        from sklearn.svm import SVC

        return SVC(
            probability=True,
            kernel="rbf",
            C=5.0,
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        ), "SVM-RBF"

    def make_rf():
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ), "RandomForest"

    if choice == "lightgbm":
        clf, name = make_lightgbm()
    elif choice == "xgboost":
        clf, name = make_xgboost()
    elif choice == "svm":
        clf, name = make_svm()
    elif choice == "random_forest":
        clf, name = make_rf()
    else:
        clf, name = make_lightgbm()

    LOGGER.info("Training ML model: %s | X_train shape=%s", name, X_train.shape)
    clf.fit(X_train, y_train)
    return clf, name


def specificity_from_confusion_matrix(cm: np.ndarray) -> List[float]:
    cm = np.asarray(cm)
    k = cm.shape[0]
    total = cm.sum()
    specs: List[float] = []
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else 0.0)
    return specs


def evaluate_split(clf, X: pd.DataFrame, y: pd.Series, class_names: List[str], split_name: str):
    y_pred = clf.predict(X)
    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X)
        except Exception:
            y_prob = None

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=list(range(len(class_names))))

    specificity = specificity_from_confusion_matrix(cm)
    macro_specificity = float(np.mean(specificity)) if specificity else 0.0

    ppv = precision.astype(float)
    cm_np = np.asarray(cm)
    k = cm_np.shape[0]
    total = cm_np.sum()
    npv = np.zeros(k, dtype=float)
    for i in range(k):
        tp = cm_np[i, i]
        fp = cm_np[:, i].sum() - tp
        fn = cm_np[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = tn + fn
        npv[i] = float(tn / denom) if denom > 0 else 0.0

    macro_ppv = float(np.mean(ppv)) if ppv.size > 0 else 0.0
    macro_npv = float(np.mean(npv)) if npv.size > 0 else 0.0

    LOGGER.info(
        "%s | Acc=%.4f | Macro-F1=%.4f | Macro-Spec=%.4f",
        split_name,
        acc,
        macro_f1,
        macro_specificity,
    )
    for idx, cname in enumerate(class_names):
        LOGGER.info(
            "%s | %s | P=%.4f R=%.4f Spec=%.4f F1=%.4f PPV=%.4f NPV=%.4f n=%d",
            split_name,
            cname,
            precision[idx],
            recall[idx],
            specificity[idx],
            f1[idx],
            ppv[idx],
            npv[idx],
            support[idx],
        )

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_specificity": float(macro_specificity),
        "macro_ppv": float(macro_ppv),
        "macro_npv": float(macro_npv),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_specificity": [float(x) for x in specificity],
        "per_class_f1": f1.tolist(),
        "per_class_ppv": [float(x) for x in ppv.tolist()],
        "per_class_npv": [float(x) for x in npv.tolist()],
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_pred, y_prob


def save_val_roc_auc(
    y_true: pd.Series,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Path,
    auc_ci: Optional[Dict] = None,
):
    if y_prob is None:
        LOGGER.warning("Validation probabilities are None; ROC/AUC is skipped.")
        return None

    y_true = np.asarray(y_true)
    if y_prob.ndim != 2 or y_prob.shape[0] != len(y_true) or y_prob.shape[1] != len(class_names):
        LOGGER.warning("Probability shape mismatch. ROC/AUC is skipped.")
        return None

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        LOGGER.warning("Validation labels contain only one class; ROC/AUC is undefined.")
        return None

    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc_vals = {}, {}, {}
    valid_classes = []

    for i in range(len(class_names)):
        if np.unique(y_true_bin[:, i]).size < 2:
            LOGGER.warning("Class %s is missing in validation. Skipping class ROC.", class_names[i])
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])
        valid_classes.append(i)

    if not valid_classes:
        LOGGER.warning("No valid classes for ROC/AUC computation.")
        return None

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in valid_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_vals["macro"] = auc(fpr["macro"], tpr["macro"])

    def fmt_auc_with_ci(name: str, auc_value: float) -> str:
        if auc_ci is None:
            return f"{auc_value:.3f}"
        if name in ["macro", "micro"]:
            ci = auc_ci.get(name, None)
        else:
            ci = auc_ci.get("per_class", {}).get(name, None)
        if ci is None or ci[0] is None:
            return f"{auc_value:.3f}"
        return f"{auc_value:.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f})"

    plt.figure(figsize=(10, 8), dpi=300)
    lw = 2

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-average ROC (AUC = {fmt_auc_with_ci('micro', roc_auc_vals['micro'])})",
        color="deeppink",
        linestyle=":",
        linewidth=3,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average ROC (AUC = {fmt_auc_with_ci('macro', roc_auc_vals['macro'])})",
        color="navy",
        linestyle=":",
        linewidth=3,
    )

    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    for i, color in zip(valid_classes, colors):
        cname = class_names[i]
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            alpha=0.8,
            label=f"ROC class: {cname} (AUC = {fmt_auc_with_ci(cname, roc_auc_vals[i])})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, alpha=0.5)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-class ROC Curve Analysis", fontsize=14, fontweight="bold")

    if auc_ci is not None and auc_ci.get("macro") and auc_ci.get("micro"):
        macro_ci = auc_ci["macro"]
        micro_ci = auc_ci["micro"]
        if macro_ci[0] is not None and micro_ci[0] is not None:
            bottom_text = (
                f"Macro AUC = {roc_auc_vals['macro']:.3f} (95%CI {macro_ci[0]:.3f}-{macro_ci[1]:.3f})    "
                f"Micro AUC = {roc_auc_vals['micro']:.3f} (95%CI {micro_ci[0]:.3f}-{micro_ci[1]:.3f})"
            )
            plt.gcf().text(0.5, 0.01, bottom_text, ha="center", va="bottom", fontsize=9)
            plt.tight_layout(rect=[0, 0.04, 1, 1])
        else:
            plt.tight_layout()
    else:
        plt.tight_layout()

    plt.legend(loc="lower right", fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    LOGGER.info(
        "Saved validation ROC to %s | Macro AUC=%.4f | Micro AUC=%.4f",
        output_path,
        roc_auc_vals["macro"],
        roc_auc_vals["micro"],
    )

    per_class_auc = [float(roc_auc_vals[i]) if i in roc_auc_vals else None for i in range(len(class_names))]
    return {
        "macro_auc": float(roc_auc_vals["macro"]),
        "micro_auc": float(roc_auc_vals["micro"]),
        "per_class_auc": per_class_auc,
    }


def save_metrics(metrics: Dict[str, Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved metrics JSON: %s", output_path)


def save_feature_importance(clf, feature_cols: List[str], output_path: Path) -> None:
    if not hasattr(clf, "feature_importances_"):
        LOGGER.warning("Current model does not expose feature_importances_. Skipping CSV export.")
        return

    importances = getattr(clf, "feature_importances_")
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    LOGGER.info("Top 15 feature importance:\n%s", fi_df.head(15).to_string(index=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved feature importance CSV: %s", output_path)


def write_back_predictions_excel(
    df_all: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_pred_train: np.ndarray,
    y_pred_val: np.ndarray,
    class_names_print: List[str],
    output_excel_path: Path,
) -> None:
    df_out = df_all.copy()

    split = df_out["split"].astype(str).str.strip().str.lower()
    train_mask = split == "train"
    val_mask = split.isin(["val", "valid", "validation", "test"])

    df_out["ml_pred_label"] = ""
    df_out["ml_is_correct"] = ""

    n_tr = min(int(train_mask.sum()), len(y_pred_train), len(y_train))
    if n_tr > 0:
        tr_idx = df_out.index[train_mask][:n_tr]
        df_out.loc[tr_idx, "ml_pred_label"] = [class_names_print[i] for i in y_pred_train[:n_tr]]
        df_out.loc[tr_idx, "ml_is_correct"] = (y_pred_train[:n_tr] == y_train.values[:n_tr]).tolist()

    n_va = min(int(val_mask.sum()), len(y_pred_val), len(y_val))
    if n_va > 0:
        va_idx = df_out.index[val_mask][:n_va]
        df_out.loc[va_idx, "ml_pred_label"] = [class_names_print[i] for i in y_pred_val[:n_va]]
        df_out.loc[va_idx, "ml_is_correct"] = (y_pred_val[:n_va] == y_val.values[:n_va]).tolist()

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(output_excel_path, index=False)
    LOGGER.info("Saved predictions Excel: %s", output_excel_path)


def bootstrap_ci95_and_mean_roc(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
    roc_grid_points: int = 201,
    print_per_class: bool = True,
    save_per_class_csv: bool = True,
):
    if y_prob is None:
        LOGGER.warning("y_prob is None; bootstrap ROC/AUC CI is skipped.")
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    k = len(class_names)
    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0] or y_prob.shape[1] != k:
        LOGGER.warning("Probability shape mismatch; bootstrap CI is skipped.")
        return None

    rng = np.random.RandomState(seed)

    idx_by_class = []
    for c in range(k):
        idx_c = np.where(y_true == c)[0]
        if idx_c.size == 0:
            LOGGER.warning("Validation set has 0 samples for class '%s'.", class_names[c])
        idx_by_class.append(idx_c)

    acc_list = np.zeros(n_bootstrap, dtype=float)
    macro_f1_list = np.zeros(n_bootstrap, dtype=float)
    macro_spec_list = np.zeros(n_bootstrap, dtype=float)
    macro_auc_list = np.full(n_bootstrap, np.nan, dtype=float)
    micro_auc_list = np.full(n_bootstrap, np.nan, dtype=float)

    prec_mat = np.zeros((n_bootstrap, k), dtype=float)
    rec_mat = np.zeros((n_bootstrap, k), dtype=float)
    f1_mat = np.zeros((n_bootstrap, k), dtype=float)
    spec_mat = np.zeros((n_bootstrap, k), dtype=float)
    ppv_mat = np.zeros((n_bootstrap, k), dtype=float)
    npv_mat = np.zeros((n_bootstrap, k), dtype=float)
    auc_mat = np.full((n_bootstrap, k), np.nan, dtype=float)

    grid = np.linspace(0.0, 1.0, int(roc_grid_points))
    tpr_sum_per_class = np.zeros((k, grid.size), dtype=float)
    tpr_sum_micro = np.zeros(grid.size, dtype=float)
    tpr_sum_macro = np.zeros(grid.size, dtype=float)

    for b in tqdm(range(n_bootstrap), desc="Bootstrap 95%CI (Val)", ncols=100):
        boot_parts = []
        for c in range(k):
            idx_c = idx_by_class[c]
            if idx_c.size == 0:
                continue
            boot_parts.append(rng.choice(idx_c, size=idx_c.size, replace=True))
        if not boot_parts:
            LOGGER.warning("Bootstrap sampling produced empty indices. Abort bootstrap.")
            return None

        boot_idx = np.concatenate(boot_parts)
        rng.shuffle(boot_idx)

        yt = y_true[boot_idx]
        yp = y_pred[boot_idx]
        ypb = y_prob[boot_idx]

        acc_list[b] = accuracy_score(yt, yp)
        macro_f1_list[b] = f1_score(yt, yp, average="macro", zero_division=0)

        precision, recall, f1v, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(k)), zero_division=0
        )
        cm = confusion_matrix(yt, yp, labels=list(range(k)))
        spec = specificity_from_confusion_matrix(cm)
        macro_spec_list[b] = float(np.mean(spec)) if spec else 0.0

        ppv = precision.astype(float)
        cm_np = np.asarray(cm)
        total = cm_np.sum()
        npv = np.zeros(k, dtype=float)
        for i in range(k):
            tp = cm_np[i, i]
            fp = cm_np[:, i].sum() - tp
            fn = cm_np[i, :].sum() - tp
            tn = total - (tp + fp + fn)
            denom = tn + fn
            npv[i] = float(tn / denom) if denom > 0 else 0.0

        prec_mat[b, :] = precision
        rec_mat[b, :] = recall
        f1_mat[b, :] = f1v
        spec_mat[b, :] = np.asarray(spec, dtype=float)
        ppv_mat[b, :] = ppv
        npv_mat[b, :] = npv

        y_true_bin = label_binarize(yt, classes=list(range(k)))
        fpr_dict = {}
        tpr_dict = {}
        valid_for_macro = []

        for i in range(k):
            if y_true_bin.shape[1] <= i:
                continue
            if np.unique(y_true_bin[:, i]).size < 2:
                fpr_i = np.array([0.0, 1.0], dtype=float)
                tpr_i = np.array([0.0, 1.0], dtype=float)
                auc_i = np.nan
            else:
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], ypb[:, i])
                auc_i = auc(fpr_i, tpr_i)

            fpr_dict[i] = fpr_i
            tpr_dict[i] = tpr_i
            valid_for_macro.append(i)
            tpr_sum_per_class[i, :] += np.interp(grid, fpr_i, tpr_i)
            auc_mat[b, i] = auc_i

        if np.unique(y_true_bin.ravel()).size < 2:
            fpr_micro = np.array([0.0, 1.0], dtype=float)
            tpr_micro = np.array([0.0, 1.0], dtype=float)
            micro_auc = np.nan
        else:
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), ypb.ravel())
            micro_auc = auc(fpr_micro, tpr_micro)

        micro_auc_list[b] = micro_auc
        tpr_sum_micro += np.interp(grid, fpr_micro, tpr_micro)

        if valid_for_macro:
            all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in valid_for_macro]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in valid_for_macro:
                mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
            mean_tpr /= len(valid_for_macro)
            macro_auc = auc(all_fpr, mean_tpr)
        else:
            macro_auc = np.nan
        macro_auc_list[b] = macro_auc

        macro_tpr_grid = np.zeros(grid.size, dtype=float)
        denom = 0
        for i in valid_for_macro:
            macro_tpr_grid += np.interp(grid, fpr_dict[i], tpr_dict[i])
            denom += 1
        macro_tpr_grid = (macro_tpr_grid / denom) if denom > 0 else macro_tpr_grid
        tpr_sum_macro += macro_tpr_grid

    def ci_nan(arr_1d: np.ndarray):
        arr = np.asarray(arr_1d, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"mean": None, "ci95": [None, None]}
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return {"mean": float(np.mean(arr)), "ci95": [float(lo), float(hi)]}

    def ci_mat_nan(mat: np.ndarray, names: List[str]):
        out = {}
        for i, n in enumerate(names):
            col = mat[:, i].astype(float)
            col = col[~np.isnan(col)]
            if col.size == 0:
                out[n] = {"mean": None, "ci95": [None, None]}
            else:
                lo, hi = np.quantile(col, [0.025, 0.975])
                out[n] = {"mean": float(np.mean(col)), "ci95": [float(lo), float(hi)]}
        return out

    prec_ci = ci_mat_nan(prec_mat, class_names)
    rec_ci = ci_mat_nan(rec_mat, class_names)
    f1_ci = ci_mat_nan(f1_mat, class_names)
    spec_ci = ci_mat_nan(spec_mat, class_names)
    ppv_ci = ci_mat_nan(ppv_mat, class_names)
    npv_ci = ci_mat_nan(npv_mat, class_names)
    auc_ci_per_class = ci_mat_nan(auc_mat, class_names)

    ci_pack = {
        "n_bootstrap": int(n_bootstrap),
        "sampling": "stratified_bootstrap(per-class n_k, with replacement; total=N_val)",
        "accuracy": ci_nan(acc_list),
        "macro_f1": ci_nan(macro_f1_list),
        "macro_specificity": ci_nan(macro_spec_list),
        "macro_auc_ovr": ci_nan(macro_auc_list),
        "micro_auc_ovr": ci_nan(micro_auc_list),
        "per_class": {
            cname: {
                "precision": prec_ci[cname],
                "recall": rec_ci[cname],
                "f1": f1_ci[cname],
                "specificity": spec_ci[cname],
                "ppv": ppv_ci[cname],
                "npv": npv_ci[cname],
                "auc_ovr": auc_ci_per_class[cname],
            }
            for cname in class_names
        },
    }

    mean_tpr_per_class = tpr_sum_per_class / float(n_bootstrap)
    mean_tpr_micro = tpr_sum_micro / float(n_bootstrap)
    mean_tpr_macro = tpr_sum_macro / float(n_bootstrap)

    mean_auc_on_grid_per_class = [float(auc(grid, mean_tpr_per_class[i])) for i in range(k)]
    mean_auc_on_grid_micro = float(auc(grid, mean_tpr_micro))
    mean_auc_on_grid_macro = float(auc(grid, mean_tpr_macro))

    macro_auc_ci = ci_pack["macro_auc_ovr"]
    micro_auc_ci = ci_pack["micro_auc_ovr"]

    mean_roc_path = output_dir / "val_roc_curve_bootstrap_mean.png"
    plt.figure(figsize=(10, 8), dpi=300)
    lw = 2

    def auc_label_from_ci(fallback_auc: float, ci_obj: Dict) -> str:
        if ci_obj["mean"] is None:
            return f"{fallback_auc:.3f}"
        lo, hi = ci_obj["ci95"]
        return f"{ci_obj['mean']:.3f} (95%CI {lo:.3f}-{hi:.3f})"

    plt.plot(
        grid,
        mean_tpr_micro,
        label=f"Micro-average Mean ROC (AUC = {auc_label_from_ci(mean_auc_on_grid_micro, micro_auc_ci)})",
        color="deeppink",
        linestyle=":",
        linewidth=3,
    )
    plt.plot(
        grid,
        mean_tpr_macro,
        label=f"Macro-average Mean ROC (AUC = {auc_label_from_ci(mean_auc_on_grid_macro, macro_auc_ci)})",
        color="navy",
        linestyle=":",
        linewidth=3,
    )

    colors = cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    for i, color in zip(range(k), colors):
        cname = class_names[i]
        auc_ci_obj = ci_pack["per_class"][cname]["auc_ovr"]
        plt.plot(
            grid,
            mean_tpr_per_class[i],
            color=color,
            lw=lw,
            alpha=0.85,
            label=f"Mean ROC class: {cname} (AUC = {auc_label_from_ci(mean_auc_on_grid_per_class[i], auc_ci_obj)})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, alpha=0.5)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Bootstrap Mean ROC Curve (Val)", fontsize=14, fontweight="bold")

    if macro_auc_ci["mean"] is not None and micro_auc_ci["mean"] is not None:
        bottom_text = (
            f"Macro AUC = {macro_auc_ci['mean']:.3f} (95%CI {macro_auc_ci['ci95'][0]:.3f}-{macro_auc_ci['ci95'][1]:.3f})    "
            f"Micro AUC = {micro_auc_ci['mean']:.3f} (95%CI {micro_auc_ci['ci95'][0]:.3f}-{micro_auc_ci['ci95'][1]:.3f})"
        )
        plt.gcf().text(0.5, 0.01, bottom_text, ha="center", va="bottom", fontsize=9)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        plt.tight_layout()

    plt.legend(loc="lower right", fontsize=8.5, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(mean_roc_path, bbox_inches="tight")
    plt.close()

    ci_pack["mean_roc_plot"] = str(mean_roc_path)
    auc_ci_for_plot = {
        "macro": ci_pack["macro_auc_ovr"]["ci95"],
        "micro": ci_pack["micro_auc_ovr"]["ci95"],
        "per_class": {c: ci_pack["per_class"][c]["auc_ovr"]["ci95"] for c in class_names},
    }
    ci_pack["auc_ci_for_plot"] = auc_ci_for_plot

    LOGGER.info(
        "Bootstrap summary | Accuracy=%.4f | Macro-F1=%.4f | Macro-Spec=%.4f | Macro-AUC=%.4f | Micro-AUC=%.4f",
        ci_pack["accuracy"]["mean"] if ci_pack["accuracy"]["mean"] is not None else float("nan"),
        ci_pack["macro_f1"]["mean"] if ci_pack["macro_f1"]["mean"] is not None else float("nan"),
        ci_pack["macro_specificity"]["mean"] if ci_pack["macro_specificity"]["mean"] is not None else float("nan"),
        ci_pack["macro_auc_ovr"]["mean"] if ci_pack["macro_auc_ovr"]["mean"] is not None else float("nan"),
        ci_pack["micro_auc_ovr"]["mean"] if ci_pack["micro_auc_ovr"]["mean"] is not None else float("nan"),
    )
    LOGGER.info("Saved bootstrap mean ROC plot: %s", mean_roc_path)

    if print_per_class:
        for cname in class_names:
            pc = ci_pack["per_class"][cname]
            LOGGER.info(
                "Bootstrap per-class | %s | P=%s | R=%s | F1=%s | Spec=%s | PPV=%s | NPV=%s | AUC=%s",
                cname,
                pc["precision"],
                pc["recall"],
                pc["f1"],
                pc["specificity"],
                pc["ppv"],
                pc["npv"],
                pc["auc_ovr"],
            )

    if save_per_class_csv:
        rows = []
        for cname in class_names:
            pc = ci_pack["per_class"][cname]
            for metric_name in ["precision", "recall", "f1", "specificity", "ppv", "npv", "auc_ovr"]:
                rows.append(
                    {
                        "class": cname,
                        "metric": metric_name,
                        "mean": pc[metric_name]["mean"],
                        "ci95_low": pc[metric_name]["ci95"][0],
                        "ci95_high": pc[metric_name]["ci95"][1],
                    }
                )
        df_ci = pd.DataFrame(rows)
        csv_path = output_dir / "val_bootstrap_ci95_per_class.csv"
        df_ci.to_csv(csv_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved bootstrap per-class CI CSV: %s", csv_path)

    return ci_pack


def validate_input_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"split", "class_name", "image_path"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Excel is missing required columns: {sorted(missing)}")

    missing_clinical = [col for col in CLINICAL_FEATURE_COLS if col not in df.columns]
    if missing_clinical:
        raise ValueError(
            "Excel is missing clinical feature columns required by this script: "
            f"{missing_clinical}"
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = Path(args.excel_path)
    cnn_ckpt = Path(args.cnn_ckpt)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not cnn_ckpt.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {cnn_ckpt}")

    ml_model_path = Path(args.ml_model_path) if args.ml_model_path else (output_dir / "ml_fusion_bundle.pkl")

    if args.overwrite_input_excel:
        predictions_excel_path = excel_path
    elif args.predictions_excel_path is not None:
        predictions_excel_path = Path(args.predictions_excel_path)
    else:
        predictions_excel_path = output_dir / "predictions_with_ml.xlsx"

    df = pd.read_excel(excel_path)
    validate_input_dataframe(df)
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    model = build_cnn_model(num_classes=4, drop_rate=args.drop_rate).to(device)
    idx_to_label = load_cnn_checkpoint(model, cnn_ckpt)
    LOGGER.info("Checkpoint label mapping: %s", idx_to_label)

    LOGGER.info("Running CNN inference to extract embeddings and probabilities.")
    df, embeddings, probs = run_cnn_inference_and_extract(
        df=df,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        pca,
        _,
        _,
    ) = prepare_features_with_pca(
        df=df,
        embeddings=embeddings,
        cnn_probs=probs,
        n_components=args.pca_components,
    )

    clf, model_name = train_ml_model(X_train, y_train, args.seed, args.ml_model)

    class_names_print = ["normal", "LSIL", "HSIL", "Cancer"]
    metrics_train, y_pred_train, _ = evaluate_split(clf, X_train, y_train, class_names_print, "Train")
    metrics_val, y_pred_val, y_prob_val = evaluate_split(clf, X_val, y_val, class_names_print, "Val")

    ci_pack = bootstrap_ci95_and_mean_roc(
        y_true=y_val,
        y_pred=y_pred_val,
        y_prob=y_prob_val,
        class_names=class_names_print,
        output_dir=output_dir,
        n_bootstrap=int(args.bootstrap_iters),
        seed=args.seed,
        roc_grid_points=int(args.bootstrap_roc_grid_points),
        print_per_class=True,
        save_per_class_csv=True,
    )
    if ci_pack is not None:
        metrics_val["bootstrap_ci95"] = ci_pack

    roc_output_path = output_dir / "val_roc_curve.png"
    auc_ci_for_plot = ci_pack["auc_ci_for_plot"] if ci_pack is not None and "auc_ci_for_plot" in ci_pack else None
    roc_info = save_val_roc_auc(
        y_true=y_val,
        y_prob=y_prob_val,
        class_names=class_names_print,
        output_path=roc_output_path,
        auc_ci=auc_ci_for_plot,
    )
    if roc_info is not None:
        metrics_val["roc_auc_macro_ovr"] = roc_info["macro_auc"]
        metrics_val["roc_auc_micro_ovr"] = roc_info["micro_auc"]
        metrics_val["roc_auc_per_class"] = roc_info["per_class_auc"]

    metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "ml_model_name": model_name,
        "features": feature_names,
        "pca_components": args.pca_components,
        "bootstrap_iters": int(args.bootstrap_iters),
    }
    save_metrics(metrics, output_dir / "metrics_fusion.json")

    write_back_predictions_excel(
        df_all=df,
        y_train=y_train,
        y_val=y_val,
        y_pred_train=y_pred_train,
        y_pred_val=y_pred_val,
        class_names_print=class_names_print,
        output_excel_path=predictions_excel_path,
    )

    save_feature_importance(clf, feature_names, output_dir / "feature_importance.csv")

    bundle = {
        "clf": clf,
        "pca": pca,
        "feature_names": feature_names,
        "clinical_cols": CLINICAL_FEATURE_COLS,
        "cnn_basic_cols": CNN_BASIC_FEATURE_COLS,
        "canonical_classes": CANONICAL_CLASSES,
    }
    joblib.dump(bundle, ml_model_path)
    LOGGER.info("Saved ML fusion bundle (%s): %s", model_name, ml_model_path)


if __name__ == "__main__":
    if os.name == "nt":
        import multiprocessing
        multiprocessing.freeze_support()
    main()
