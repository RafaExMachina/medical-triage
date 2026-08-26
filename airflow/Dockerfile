FROM apache/airflow:3.3.1-python3.12

COPY --chown=airflow:root airflow/requirements.txt /requirements.txt

RUN pip install \
    --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /requirements.txt

COPY --chown=airflow:root src /opt/airflow/project/src

COPY --chown=airflow:root \
    airflow/dags \
    /opt/airflow/dags

ENV PYTHONPATH="/opt/airflow/project/src"