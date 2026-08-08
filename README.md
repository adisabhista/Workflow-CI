# Workflow-CI

Repositori **Kriteria 3** submission kelas *Membangun Sistem Machine Learning* (Dicoding).
Berisi MLflow Project untuk re-training otomatis model prediksi churn, workflow CI
GitHub Actions, penyimpanan artefak, serta pembuatan Docker Image ke Docker Hub.

## Struktur Repository

```
Workflow-CI
├── .github/
│   └── workflows/
│       └── ci.yml                      # Workflow CI (Basic + Skilled + Advance)
├── MLProject/
│   ├── MLProject                       # Definisi MLflow Project
│   ├── conda.yaml                      # Environment conda
│   ├── python_env.yaml                 # Environment alternatif (virtualenv)
│   ├── requirements.txt
│   ├── modelling.py                    # Entry point training
│   ├── DockerHub.txt                   # Tautan Docker Hub
│   └── telco_preprocessing/            # Dataset siap dilatih
│       ├── telco_churn_train.csv
│       ├── telco_churn_test.csv
│       └── metadata.json
└── README.md
```

## Tahapan Workflow CI

| # | Step | Tingkat |
|---|---|---|
| 1 | Run actions/checkout@v3 | Basic |
| 2 | Set up Python 3.12.7 | Basic |
| 3 | Check Env | Basic |
| 4 | Install dependencies | Basic |
| 5 | **Run mlflow project** | Basic |
| 6 | **Get latest MLflow run_id** | Basic |
| 7 | Show model metrics | Basic |
| 8 | Install Python dependencies | Skilled |
| 9 | **Upload artefak ke GitHub Actions Artifacts** | Skilled |
| 10 | **Upload to GitHub** (commit artefak ke repo) | Skilled |
| 11 | **Build Docker Model** (`mlflow models build-docker`) | Advance |
| 12 | Log in to Docker Hub | Advance |
| 13 | Tag Docker Image | Advance |
| 14 | **Push Docker Image** | Advance |
| 15 | Complete job | — |

## Trigger

- push ke branch `main`
- pull request ke `main` (training + artefak saja, tanpa push Docker)
- manual via `workflow_dispatch` (bisa mengubah `n_estimators` & `max_depth`)
- terjadwal setiap Senin 01:00 UTC

## Secrets yang Diperlukan

Atur pada **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|---|---|
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access token Docker Hub (bukan password) |

## Menjalankan Secara Lokal

```bash
pip install -r MLProject/requirements.txt

# Menjalankan MLflow Project
mlflow run MLProject --env-manager=local \
    -P n_estimators=200 -P max_depth=12 -P class_weight=balanced

# Melihat hasil
mlflow ui
```

## Membangun Docker Image Secara Lokal

```bash
RUN_ID=$(cat MLProject/run_id.txt)
mlflow models build-docker --model-uri "runs:/$RUN_ID/model" \
    --name telco-churn-model --env-manager local

docker run -p 5005:8080 telco-churn-model
```

> **`--env-manager local` wajib.** Dengan default (`virtualenv`), MLflow mengalihkan
> base image ke `ubuntu:20.04` yang meng-install `python3.9` lalu mengunduh
> `get-pip.py` versi umum — dan skrip itu sekarang menolak Python < 3.10, sehingga
> build gagal. Dengan `local`, base image tetap `python:3.12.7-slim` sesuai
> `python_env.yaml` model.
