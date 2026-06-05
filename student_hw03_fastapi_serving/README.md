# HW03 — Serving Your HW02 Model with FastAPI and Swagger

## Goal

Build a FastAPI service that serves the model you trained and tracked in HW02.

This homework is not another notebook. The goal is to expose your model through an API and test it through Swagger.

## Required endpoints

You must implement:

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`

Swagger must be available at:

```text
http://127.0.0.1:8000/docs
```

## Setup on Windows CMD

Create and activate a clean environment:

```cmd
python -m venv .venv_hw03
.venv_hw03\Scriptsctivate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set your MLflow credentials:

```cmd
set MLFLOW_TRACKING_URI=http://185.50.38.163:33014
set MLFLOW_TRACKING_USERNAME=student_your_username
set MLFLOW_TRACKING_PASSWORD=your_mlflow_password
set STUDENT_USERNAME=student_your_username
set MLFLOW_EXPERIMENT_NAME=qbc12_hw02_student_your_username
```

If auto-selection fails, set your best HW02 run id manually:

```cmd
set MLFLOW_RUN_ID=your_hw02_run_id
```

Run the API:

```cmd
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## Your TODOs

1. Complete model loading from MLflow in `app/model_loader.py`.
2. Complete feature validation in `app/predictor.py`.
3. Complete probability/prediction logic in `app/predictor.py`.
4. Test all endpoints in Swagger.
5. Submit required screenshots under `screenshots/`.
6. Write down your selected HW02 run ID in your report or README.

## Required Swagger tests

Use the JSON files in `data/`:

- `valid_predict_request.json`
- `valid_batch_request.json`
- `bad_request_missing_field.json`
- `bad_request_wrong_type.json`
- `bad_request_leakage_field.json`

You must show both successful and failed requests.

## Forbidden model inputs

The prediction API must not accept these fields:

```text
listing_id
cutoff_date
dataset_version
future_calendar_days_observed_30d
future_available_days_30d
future_available_rate_30d
high_demand_proxy
```

These are audit, future-window, or target columns. Accepting them is leakage.

## Submission

Submit the whole folder with:

- `app/`
- `data/`
- `screenshots/`
- `requirements.txt`
- your README/report

## Hard grading penalties

- No Swagger screenshots: max 60%.
- No working `/predict`: max 50%.
- Accepting leakage fields: max 50%.
- Only notebook, no FastAPI app: max 60%.
