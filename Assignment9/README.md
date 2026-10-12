# Assignment 9: ML Pipeline Orchestration with Apache Airflow

This assignment automates a full ML workflow with an Apache Airflow DAG (Directed Acyclic Graph) running in Docker.

**Pipeline:** `load_data → clean_data → split_data → save_data → train_model → evaluate_model`

## Results
- All 6 tasks succeeded on the first try. The whole run took 11.5 seconds.
- Cleaning removed 1 duplicate row (150 → 149 rows).
- The split gave 119 training rows and 30 test rows.
- Logistic Regression scored a test accuracy of **0.967** and a macro F1-score of 0.967.

## Files
- `dags/iris_ml_pipeline.py`: the DAG with all six tasks
- `start_airflow.sh`: starts Airflow standalone in Docker, with the `dags/` and `data/` folders mounted
- `data/`: the files the pipeline wrote (CSV files, `model.pkl`, `metrics.json`)
- `Assignment9_Airflow_ML_Pipeline_Report.pdf`: the workflow write-up with screenshots
- `Assignment9_Hiren_Patel.zip`: the submission package (DAG, start script and PDF)

## How to run
1. Start Docker Desktop.
2. In Git Bash, inside this folder, run `bash start_airflow.sh`.
3. Get the admin password with `docker logs airflow 2>&1 | grep -i password`.
4. Open http://localhost:8080, find `iris_ml_pipeline` and click **Trigger**.
