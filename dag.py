from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import yaml
import pandas as pd
import logging
from modules.logging import get_logging

default_args = {
    'owner': 'gerasimov',
    'start_date': datetime(2026, 03, 01),
    'retries': 0
}

dag = DAG(
    'data_quality_checks',
    default_args=default_args,
    schedule_interval='*/30 * * * *',
    catchup=False
)

def run_check(project, parent_project, check, check_path, logical_date, **context):
    import os
    import yaml
    import pandas as pd
    from datetime import datetime, timedelta
    from airflow.exceptions import AirflowSkipException, AirflowFailException
    from checker.DataQualityCheck import DataQualityCheck

    config_file_path = os.path.join(check_path, 'config.yml')
    if not os.path.exists(config_file_path):
        raise AirflowFailException(f"Missing config for {check}")

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    source_sql, target_sql = None, None
    try:
        with open(os.path.join(check_path, config['queries']['source']['query']), 'r') as f:
            source_sql = f.read()
        if 'target' in config['queries']:
            with open(os.path.join(check_path, config['queries']['target']['query']), 'r') as f:
                target_sql = f.read()
    except Exception as e:
        logging.warning(f"SQL file error: {e}")

    dq_check = DataQualityCheck(
        project=project,
        parent_project=parent_project or None,
        config=config,
        source_sql=source_sql,
        target_sql=target_sql
    )
    
    from dateutil import parser
    logical_date_dt = parser.parse(logical_date)
    
    current_time = pd.Timestamp(logical_date_dt) + timedelta(minutes=30)
    interval_start = pd.Timestamp(logical_date_dt)
    
    status = dq_check.should_run(interval_start, current_time)
    
    if status == 'wrong':
        raise AirflowFailException(f"Cron error in {check}")
    elif status == 'skip':
        raise AirflowSkipException(f"Skipping {check} by schedule")
    
    dq_check.run_check()

file_path = os.path.dirname(os.path.abspath(file))
projects_dir = os.path.join(file_path, 'projects')

if os.path.exists(projects_dir):
    for project in os.listdir(projects_dir):
        project_path = os.path.join(projects_dir, project)
        if not os.path.isdir(project_path): continue

        with TaskGroup(group_id=f"{project}_project") as project_group:
            for check in os.listdir(project_path):
                check_path = os.path.join(project_path, check)
                if not os.path.isdir(check_path): continue

                sub_checks = [d for d in os.listdir(check_path) if os.path.isdir(os.path.join(check_path, d))]
                
                if sub_checks:
                    for sub in sub_checks:
                        sub_path = os.path.join(check_path, sub)
                        task = PythonOperator(
                            task_id=f"{project}_{check}_{sub}_run",
                            python_callable=run_check,
                            op_kwargs={
                                'project': check,
                                'parent_project': project,
                                'check': sub,
                                'check_path': sub_path,
                                'logical_date': "{{ logical_date }}"
                            }
                        )
                        if os.path.exists(os.path.join(sub_path, 'sensor.sql')):
                            sc, sn = get_sensor_tasks(sub_path, project, sub, parent_project=check)
                            sc >> sn >> task
                else:
                    task = PythonOperator(
                        task_id=f"{project}_{check}_run",
                        python_callable=run_check,
                        op_kwargs={
                            'project': project,
                            'parent_project': '',
                            'check': check,
                            'check_path': check_path,
                            'logical_date': "{{ logical_date }}"
                        }
                    )
                    if os.path.exists(os.path.join(check_path, 'sensor.sql')):
                        sc, sn = get_sensor_tasks(check_path, project, check)
                        sc >> sn >> task
