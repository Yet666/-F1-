from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    hidden_dim: int = 160
    num_blocks: int = 4
    dropout: float = 0.15
    drop_path_rate: float = 0.05
    use_se: bool = True
    se_reduction: int = 8
    use_glu: bool = True
    glu_dim: int = 64


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    early_stop_patience: int = 8
    n_folds: int = 5
    random_seed: int = 3407


@dataclass
class BlendConfig:
    meta_learner_alpha: float = 0.01


class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        squeezed = dim // reduction
        self.gate = nn.Sequential(
            nn.Linear(dim, squeezed, bias=False),
            nn.SiLU(),
            nn.Linear(squeezed, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * self.gate(x).sigmoid()


class PreActResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_se: bool = True, se_reduction: int = 8):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.se = SqueezeExcitation(dim, se_reduction) if use_se else nn.Identity()

        nn.init.xavier_uniform_(self.linear1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.se(out)
        return out + residual


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, device=x.device)
        mask = mask.floor_()
        return x / keep_prob * mask


class F1PitRegressor(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout * 0.5),
        )

        self.glu = (
            GatedLinearUnit(cfg.hidden_dim, cfg.glu_dim)
            if cfg.use_glu
            else nn.Identity()
        )
        glu_out = cfg.glu_dim if cfg.use_glu else cfg.hidden_dim

        drop_rates = np.linspace(0, cfg.drop_path_rate, cfg.num_blocks).tolist()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                PreActResBlock(glu_out, cfg.dropout, cfg.use_se, cfg.se_reduction),
                DropPath(drop_rates[i]),
            )
            for i in range(cfg.num_blocks)
        ])

        self.head = nn.Sequential(
            nn.BatchNorm1d(glu_out),
            nn.Linear(glu_out, cfg.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.glu(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class FeatureProcessor:
    def __init__(self, outlier_clip: float = 0.995):
        self.outlier_clip = outlier_clip
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.clip_bounds: Dict[str, Tuple[float, float]] = {}

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        self._detect_columns(X)

        if self.outlier_clip > 0:
            for col in self.numerical_cols:
                lo = X[col].quantile(1 - self.outlier_clip)
                hi = X[col].quantile(self.outlier_clip)
                self.clip_bounds[col] = (lo, hi)

        X = self._clip_outliers(X)

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("robust", RobustScaler()),
            ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)),
        ])

        transformers = []
        if self.numerical_cols:
            transformers.append(("num", num_pipe, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", n_jobs=-1)
        return self.preprocessor.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._clip_outliers(X)
        return self.preprocessor.transform(X)

    def _detect_columns(self, X: pd.DataFrame):
        self.categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.numerical_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.clip_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, cfg: TrainConfig, verbose: bool = True):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.verbose = verbose
        self.best_state: Optional[dict] = None
        self.best_score = float("inf")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        train_loader = self._to_loader(X_train, y_train, shuffle=True)
        val_loader = self._to_loader(X_val, y_val, shuffle=False)

        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.cfg.scheduler_t0, T_mult=self.cfg.scheduler_t_mult,
            eta_min=self.cfg.learning_rate * 1e-3,
        )
        criterion = nn.L1Loss()
        early_stop = EarlyStopping(self.cfg.early_stop_patience)
        best_state = None
        best_score = float("inf")

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx).squeeze(-1), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            val_mae = self._evaluate(val_loader)

            if val_mae < best_score:
                best_score = val_mae
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            improved = early_stop.step(val_mae)

            if self.verbose:
                lr_now = optimizer.param_groups[0]["lr"]
                flag = " ✓" if improved else ""
                print(f"  Epoch {epoch + 1:02d} | Train Loss: {train_loss / len(train_loader):.5f} | "
                      f"Val MAE: {val_mae:.5f} | LR: {lr_now:.2e}{flag}")

            if early_stop.should_stop:
                if self.verbose:
                    print(f"  >>> Early stop at epoch {epoch + 1}, best MAE: {best_score:.5f}")
                break

        self.best_state = best_state
        self.best_score = best_score
        self.model.load_state_dict(best_state)
        return best_score

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        self.model.eval()

        loader = self._to_loader(X, None, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                preds.append(self.model(bx.to(self.device)).squeeze(-1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in loader:
                all_preds.append(self.model(bx.to(self.device)).squeeze(-1).cpu().numpy())
                all_targets.append(by.numpy())
        return mean_absolute_error(np.concatenate(all_targets), np.concatenate(all_preds))

    def _to_loader(self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool) -> DataLoader:
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y, dtype=torch.float32))
        return DataLoader(TensorDataset(*tensors), batch_size=self.cfg.batch_size,
                          shuffle=shuffle, pin_memory=True, drop_last=shuffle)


def train_with_kfold(
    X: np.ndarray, y: np.ndarray, model_factory,
    device: torch.device, train_cfg: TrainConfig, model_cfg: ModelConfig,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    kf = KFold(n_splits=train_cfg.n_folds, shuffle=True, random_state=train_cfg.random_seed)
    oof_preds_list, oof_targets_list, fold_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"\n{'='*50}\n  Fold {fold + 1}/{train_cfg.n_folds}\n{'='*50}")

        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = model_factory()
        trainer = Trainer(model, device, train_cfg, verbose=verbose)
        score = trainer.fit(X_tr, y_tr, X_va, y_va)

        oof_preds_list.append(trainer.predict(X_va))
        oof_targets_list.append(y_va)
        fold_scores.append(score)

        if verbose:
            print(f"  Fold {fold + 1} best MAE: {score:.5f}")

    if verbose:
        print(f"\n  K-Fold avg MAE: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    return oof_preds_list, oof_targets_list, fold_scores


class RidgeBlender:
    def __init__(self, alpha: float = 0.01, non_negative: bool = True):
        self.alpha = alpha
        self.non_negative = non_negative
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.model_names: List[str] = []

    def fit(self, prediction_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> "RidgeBlender":
        self.model_names = list(prediction_dict.keys())
        X_stack = np.column_stack([prediction_dict[name] for name in self.model_names])

        if self.non_negative:
            self.weights = self._non_negative_ridge(X_stack, y_true)
        else:
            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(X_stack, y_true)
            self.weights = ridge.coef_
            self.intercept = ridge.intercept_

        for name, w in zip(self.model_names, self.weights):
            print(f"  {name}: weight = {w:.4f}")
        print(f"  intercept = {self.intercept:.4f}")
        return self

    def predict(self, prediction_dict: Dict[str, np.ndarray]) -> np.ndarray:
        X_stack = np.column_stack([prediction_dict[name] for name in self.model_names])
        return X_stack @ self.weights + self.intercept

    def _non_negative_ridge(self, X: np.ndarray, y: np.ndarray, max_iter: int = 20) -> np.ndarray:
        n_features = X.shape[1]
        active = np.ones(n_features, dtype=bool)
        weights = np.zeros(n_features)

        for _ in range(max_iter):
            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(X[:, active], y)
            w_active = ridge.coef_

            if np.all(w_active >= 0):
                weights[active] = w_active
                self.intercept = ridge.intercept_
                return weights

            new_active = active.copy()
            for j, idx in enumerate(np.where(active)[0]):
                if w_active[j] < 0:
                    new_active[idx] = False
            active = new_active

            if active.sum() == 0:
                return np.ones(n_features) / n_features

        return weights


def _detect_environment() -> Tuple[Optional[Path], Optional[Path], bool]:
    kaggle_comp = Path("/kaggle/input/competitions/playground-series-s6e5")
    kaggle_blend = Path("/kaggle/input/datasets/anthonytherrien/predicting-f1-pit-stops-vault")

    if kaggle_comp.exists():
        print("Kaggle environment detected")
        return kaggle_comp, kaggle_blend, True

    env_comp = os.environ.get("F1_COMP_PATH")
    if env_comp:
        comp = Path(env_comp)
        blend = Path(os.environ.get("F1_BLEND_PATH")) if os.environ.get("F1_BLEND_PATH") else None
        if comp.exists():
            return comp, blend, False

    for candidate in [Path("data"), Path("../data"), Path("input"), Path("../input"), Path.cwd()]:
        if (candidate / "train.csv").exists():
            blend_candidate = candidate / "blend"
            return candidate, (blend_candidate if blend_candidate.exists() else None), False

    return None, None, False


def main():
    COMP_PATH, BLEND_PATH, IS_KAGGLE = _detect_environment()

    if COMP_PATH is None:
        print("Data not found. Place train.csv / test.csv in current directory or data/ subfolder.")
        return

    TARGET = "PitNextLap"
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    blend_cfg = BlendConfig()

    print("Loading data...")
    train_df = pd.read_csv(COMP_PATH / "train.csv")
    test_df = pd.read_csv(COMP_PATH / "test.csv")
    print(f"  Train: {train_df.shape[0]:,} x {train_df.shape[1]}, Test: {test_df.shape[0]:,} x {test_df.shape[1]}")

    test_ids = test_df["id"]
    X_raw = train_df.drop(columns=[TARGET, "id"])
    y_raw = train_df[TARGET].values.astype(np.float32)
    X_test_raw = test_df.drop(columns=["id"])

    print("Feature engineering...")
    processor = FeatureProcessor(outlier_clip=0.995)
    X = processor.fit_transform(X_raw, y_raw)
    X_test = processor.transform(X_test_raw)
    print(f"  Feature dim: {X.shape[1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nK-Fold training ({train_cfg.n_folds} folds)...")

    def make_model():
        return F1PitRegressor(input_dim=X.shape[1], cfg=model_cfg)

    oof_preds_list, oof_targets_list, fold_scores = train_with_kfold(
        X=X, y=y_raw, model_factory=make_model, device=device,
        train_cfg=train_cfg, model_cfg=model_cfg,
    )

    oof_preds_all = np.concatenate(oof_preds_list)
    oof_targets_all = np.concatenate(oof_targets_list)
    nn_oof_mae = mean_absolute_error(oof_targets_all, oof_preds_all)
    print(f"\nNN OOF MAE: {nn_oof_mae:.5f}")

    print("\nGenerating test predictions...")
    test_preds_folds = []
    for fold in range(train_cfg.n_folds):
        model = make_model()
        trainer = Trainer(model, device, train_cfg, verbose=False)

        fold_seed = train_cfg.random_seed + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        trainer.fit(X, y_raw, X[-1000:], y_raw[-1000:])
        test_preds_folds.append(trainer.predict(X_test))
        print(f"  Fold {fold + 1} done")

    nn_test_preds = np.mean(test_preds_folds, axis=0)

    print("\nLoading external predictions...")
    prediction_dict = {"NeuralNet_OOF": oof_preds_all}

    if BLEND_PATH is not None and BLEND_PATH.exists():
        for i, fname in enumerate(["submission.csv", "submission (1).csv"]):
            path = BLEND_PATH / fname
            if path.exists():
                prediction_dict[f"External_{i + 1}"] = pd.read_csv(path)[TARGET].values
                print(f"  Loaded External_{i + 1}: {fname}")
    else:
        print("  No external predictions found, using NN only")

    if len(prediction_dict) >= 2:
        print("\nRidge blending...")
        blender = RidgeBlender(alpha=blend_cfg.meta_learner_alpha)
        blender.fit(prediction_dict, oof_targets_all)

        final_blend_dict = {"NeuralNet": nn_test_preds}
        if BLEND_PATH is not None and BLEND_PATH.exists():
            for i, fname in enumerate(["submission.csv", "submission (1).csv"]):
                path = BLEND_PATH / fname
                if path.exists():
                    final_blend_dict[f"External_{i + 1}"] = pd.read_csv(path)[TARGET].values

        test_stack = np.column_stack([final_blend_dict[n] for n in blender.model_names])
        final_predictions = test_stack @ blender.weights + blender.intercept
    else:
        final_predictions = nn_test_preds

    output_path = COMP_PATH / "submission_optimized.csv" if IS_KAGGLE else Path("submission_optimized.csv")
    submission = pd.DataFrame({"id": test_ids, TARGET: final_predictions})
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")

    print(f"\n{'='*50}")
    print(f"K-Fold avg MAE: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"OOF MAE: {nn_oof_mae:.5f}")
    print(f"Feature dim: {X.shape[1]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
