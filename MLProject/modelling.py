"""
modelling.py  (MLflow Project entry point)
==========================================
Kriteria 3 - Workflow CI.

Skrip ini merupakan penyesuaian dari `modelling.py` pada Kriteria 2 agar dapat
dijalankan sebagai **MLflow Project** di dalam GitHub Actions. Perbedaannya:

* seluruh hyperparameter diterima sebagai parameter MLflow Project;
* tracking otomatis memakai local file store `mlruns/` (tidak butuh server);
* model di-log pada artifact path `model` sehingga siap dipakai
  `mlflow models build-docker` untuk pembuatan Docker Image;
* run_id ditulis ke `run_id.txt` agar mudah dibaca langkah berikutnya di CI.

Dijalankan oleh:
    mlflow run MLProject --env-manager=local

Author : Abhista Arief Bonianto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "Telco Churn - CI Retraining"
RANDOM_STATE = 42


def load_dataset(data_dir: Path):
    train_path = data_dir / "telco_churn_train.csv"
    test_path = data_dir / "telco_churn_test.csv"
    if not train_path.exists() or not test_path.exists():
        sys.exit(f"Dataset tidak ditemukan di '{data_dir}'.")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return (
        train_df.drop(columns=["Churn"]),
        test_df.drop(columns=["Churn"]),
        train_df["Churn"],
        test_df["Churn"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(BASE_DIR / "telco_preprocessing"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight", default="balanced")
    args = parser.parse_args()

    class_weight = None if str(args.class_weight).lower() in {"none", ""} else args.class_weight

    print("=" * 70)
    print("CI RETRAINING - TELCO CUSTOMER CHURN")
    print("=" * 70)

    # Ketika dijalankan lewat `mlflow run`, run & experiment sudah dibuat oleh
    # MLflow (run id dikirim melalui env MLFLOW_RUN_ID). Set experiment hanya
    # bila skrip dijalankan langsung (`python modelling.py`).
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI : {mlflow.get_tracking_uri()}")

    X_train, X_test, y_train, y_test = load_dataset(Path(args.data_dir))
    print(f"Data latih   : {X_train.shape}, Data uji : {X_test.shape}")

    artifact_dir = BASE_DIR / "ci_artifacts"
    artifact_dir.mkdir(exist_ok=True)

    # `mlflow.start_run()` otomatis melanjutkan run yang dibuat `mlflow run`
    # bila env MLFLOW_RUN_ID tersedia; jika tidak, run baru dibuat.
    with mlflow.start_run(run_name="RandomForest-CI") as run:
        run_id = run.info.run_id

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # ---------------- Manual logging ----------------
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": str(class_weight),
                "random_state": RANDOM_STATE,
                "n_features": X_train.shape[1],
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
            }
        )

        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "training_accuracy_score": accuracy_score(y_train, y_train_pred),
            "training_precision_score": precision_score(y_train, y_train_pred, zero_division=0),
            "training_recall_score": recall_score(y_train, y_train_pred, zero_division=0),
            "training_f1_score": f1_score(y_train, y_train_pred, zero_division=0),
            "training_log_loss": log_loss(y_train, y_train_proba),
            "training_roc_auc": roc_auc_score(y_train, y_train_proba),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1_score": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_log_loss": log_loss(y_test, y_proba),
            "test_average_precision": average_precision_score(y_test, y_proba),
        }
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # ---------------- Artefak ----------------
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1], ["Tidak Churn", "Churn"])
        ax.set_yticks([0, 1], ["Tidak Churn", "Churn"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - CI Retraining")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = artifact_dir / "training_confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['test_roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - CI Retraining")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = artifact_dir / "roc_curve.png"
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)

        metric_info = {
            "run_id": run_id,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "class_weight": str(class_weight),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Tidak Churn", "Churn"], output_dict=True
            ),
        }
        info_path = artifact_dir / "metric_info.json"
        info_path.write_text(json.dumps(metric_info, indent=2), encoding="utf-8")

        est_path = artifact_dir / "estimator.html"
        from sklearn.utils import estimator_html_repr

        est_path.write_text(estimator_html_repr(model), encoding="utf-8")

        for p in [cm_path, roc_path, info_path, est_path]:
            mlflow.log_artifact(str(p))

        # ---------------- Model ----------------
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        mlflow.set_tags(
            {
                "dataset": "Telco Customer Churn",
                "author": "Abhista Arief Bonianto",
                "criteria": "Kriteria 3 - Workflow CI",
                "trigger": os.getenv("GITHUB_EVENT_NAME", "local"),
                "commit": os.getenv("GITHUB_SHA", "-")[:8],
            }
        )

        # run_id disimpan agar mudah dibaca langkah selanjutnya pada workflow CI
        (BASE_DIR / "run_id.txt").write_text(run_id, encoding="utf-8")

        print("-" * 70)
        for k, v in metrics.items():
            print(f"  {k:<28}: {v:.4f}")
        print("-" * 70)
        print(f"MLflow run_id : {run_id}")
        print("=" * 70)


if __name__ == "__main__":
    main()
