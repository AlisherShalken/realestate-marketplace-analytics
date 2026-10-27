"""
Ежедневное обновление витрин mart.user_daily / mart.listing_daily
и проверки качества данных для активных экспериментов.

user_daily и listing_daily независимы, проверки идут после обеих.
SRM проверяем каждый день: лучше узнать о сломанном сплите на 2-й день теста, а не в конце.
"""
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryInsertJobOperator,
)
from airflow.providers.google.cloud.sensors.bigquery import BigQueryTablePartitionExistenceSensor
from scipy import stats

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
GCP_CONN = "gcp_analytics"
PROJECT = "analytics-prod"

default_args = {
    "owner": "product-analytics",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def bq_job(task_id, sql_file):
    return BigQueryInsertJobOperator(
        task_id=task_id,
        gcp_conn_id=GCP_CONN,
        configuration={"query": {"query": (SQL_DIR / sql_file).read_text(encoding="utf-8"),
                                  "useLegacySql": False}},
    )


def check_srm(ds, **_):
    """Падаем, если хотя бы в одном активном эксперименте p-value SRM < 0.001."""
    hook = BigQueryHook(gcp_conn_id=GCP_CONN, use_legacy_sql=False)
    rows = hook.get_records(f"""
        SELECT e.experiment, e.variant, COUNT(DISTINCT e.user_id) AS n
        FROM `{PROJECT}.mart.exp_exposures` e
        JOIN `{PROJECT}.dwh.experiments` x USING (experiment)
        WHERE DATE('{ds}') BETWEEN x.start_date AND x.end_date
          AND e.first_exposure_dt <= DATE('{ds}')
        GROUP BY 1, 2
    """)
    by_exp = {}
    for exp, variant, n in rows:
        by_exp.setdefault(exp, {})[variant] = n

    broken = []
    for exp, groups in by_exp.items():
        counts = list(groups.values())
        p = stats.chisquare(counts).pvalue  # сплиты у нас всегда равные
        print(f"{exp}: {groups}, SRM p={p:.4g}")
        if p < 0.001:
            broken.append(exp)
    if broken:
        raise AirflowFailException(f"SRM в экспериментах: {broken}")


with DAG(
    dag_id="product_marts_daily",
    start_date=datetime(2025, 6, 1),
    schedule="0 3 * * *",  # после того как загрузились события за вчера
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["product", "marts"],
) as dag:

    wait_events = BigQueryTablePartitionExistenceSensor(
        task_id="wait_events_partition",
        gcp_conn_id=GCP_CONN,
        project_id=PROJECT,
        dataset_id="analytics",
        table_id="events",
        partition_id="{{ ds_nodash }}",
        mode="reschedule",
        poke_interval=600,
        timeout=6 * 3600,
    )

    user_daily = bq_job("build_user_daily", "user_daily.sql")
    listing_daily = bq_job("build_listing_daily", "listing_daily.sql")

    # витрина не должна "проваливаться" больше чем на 30% к среднему за прошлую неделю
    volume_check = BigQueryCheckOperator(
        task_id="check_user_daily_volume",
        gcp_conn_id=GCP_CONN,
        use_legacy_sql=False,
        sql="""
            SELECT
              (SELECT COUNT(*) FROM `mart.user_daily` WHERE dt = '{{ ds }}')
              > 0.7 * (SELECT COUNT(*) / 7 FROM `mart.user_daily`
                       WHERE dt BETWEEN DATE_SUB('{{ ds }}', INTERVAL 7 DAY) AND DATE_SUB('{{ ds }}', INTERVAL 1 DAY))
        """,
    )

    # CR объявлений в разумных пределах — ловит дубли событий и поломки трекинга контактов
    cr_check = BigQueryCheckOperator(
        task_id="check_listing_cr_range",
        gcp_conn_id=GCP_CONN,
        use_legacy_sql=False,
        sql="""
            SELECT SAFE_DIVIDE(SUM(contacts), SUM(views)) BETWEEN 0.01 AND 0.3
            FROM `mart.listing_daily` WHERE dt = '{{ ds }}'
        """,
    )

    srm = PythonOperator(task_id="check_experiments_srm", python_callable=check_srm)

    wait_events >> [user_daily, listing_daily]
    user_daily >> volume_check
    listing_daily >> cr_check
    [volume_check, cr_check] >> srm
