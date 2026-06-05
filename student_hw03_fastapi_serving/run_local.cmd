@echo off
REM Fill these values from the MLflow credentials sheet.
set MLFLOW_TRACKING_URI=http://185.50.38.163:33014
set MLFLOW_TRACKING_USERNAME=student_your_username
set MLFLOW_TRACKING_PASSWORD=your_mlflow_password
set STUDENT_USERNAME=student_your_username
set MLFLOW_EXPERIMENT_NAME=qbc12_hw02_student_your_username
set MLFLOW_RUN_ID=
set PREDICTION_THRESHOLD=0.5

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
