#!/bin/bash
# I start Airflow in Docker with my dags and data folders mounted from this folder
# Run from Git Bash inside the Assignment9 folder:  bash start_airflow.sh
mkdir -p dags data logs
docker rm -f airflow 2>/dev/null
MSYS_NO_PATHCONV=1 docker run -d --name airflow -p 8080:8080 \
  -v "$(pwd -W)/dags:/opt/airflow/dags" \
  -v "$(pwd -W)/data:/opt/airflow/data" \
  -e _PIP_ADDITIONAL_REQUIREMENTS="scikit-learn pandas" \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
  apache/airflow:latest standalone
echo "Airflow is starting (1-3 minutes). Then open http://localhost:8080"
echo "Find the admin password with:  docker logs airflow 2>&1 | grep -i password"
