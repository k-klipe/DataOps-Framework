import os
import pandas as pd
import numpy as np
import logging
from functools import wraps
from datetime import datetime
from sklearn.ensemble import IsolationForest
from prophet import Prophet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataQualityCheck:
    def __init__(self, project, config, parent_project=None, source_sql=None, target_sql=None):
        self.config = config
        self.project = project
        self.parent_project = parent_project
        self.check_name = config.get('check_name', 'DQ_Check')
        self.check_type = config.get('check_type')
        self.queries = config.get('queries', {})
        self.threshold = config.get('threshold', 0.1)
        self.abs = config.get('abs', False)
        self.larger = config.get('larger', False)
        self.skip = config.get('skip', False)
        self.source_sql = source_sql
        self.target_sql = target_sql

    def _get_connection(self, db_type, connection_id):
        from airflow.hooks.base import BaseHook
        from clickhouse_driver import Client
        from sqlalchemy import create_engine
        from urllib.parse import quote_plus

        conn = BaseHook.get_connection(connection_id)
        
        if db_type == 'vertica':
            conn_str = f'vertica+vertica_python://{conn.login}:{quote_plus(conn.password)}@{conn.host}:{conn.port}/{conn.schema}'
            return create_engine(conn_str, pool_pre_ping=True)
        elif db_type == 'clickhouse':
            return Client(host=conn.host, user=conn.login, password=conn.password, port=conn.port)
        else:
            raise ValueError(f"Unsupported DB: {db_type}")

    def execute_query(self, db_type, connection_id, query):
        engine = self._get_connection(db_type, connection_id)
        if db_type == 'clickhouse':
            data, columns = engine.execute(query, with_column_types=True)
            return pd.DataFrame(data, columns=[col[0] for col in columns])
        else:
            return pd.read_sql(query, engine)

    def quality_check_decorator(check_function):
        @wraps(check_function)
        def wrapper(self, *args, **kwargs):
            full_df = check_function(self, *args, **kwargs)
            if self.check_type == 'table_empty':
                return full_df, full_df
            return full_df[['date', 'check_value', 'threshold', 'check']], full_df
        return wrapper

    @quality_check_decorator
    def check_percentage_deviation(self):
        dfs = {}
        for s in ['source', 'target']:
            cfg = self.queries[s]
            sql = self.source_sql if s == 'source' else self.target_sql
            df = self.execute_query(cfg['db'], cfg['connection_id'], sql)
            df.rename(columns={str(cfg['value_column']): f'{s}_value', str(cfg['date_column']): 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            dfs[s] = df

        check_df = pd.merge(dfs['source'], dfs['target'], on='date', how='outer').dropna()
        
        def calc_diff(row):
            if row['target_value'] == 0: return 0
            diff = row['source_value'] - row['target_value']
            val = abs(diff) / row['target_value'] if self.abs else diff / row['target_value']
            return round(val, 3)

        check_df['check_value'] = check_df.apply(calc_diff, axis=1)
        check_df['threshold'] = self.threshold
        check_df['check'] = check_df['check_value'].apply(lambda x: x >= self.threshold if self.larger else x <= self.threshold)
        return check_df

    @quality_check_decorator
    def check_table_empty(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        cnt = df.iloc[0, 0] if not df.empty else 0
        return pd.DataFrame({'date': [datetime.now()], 'check_value': [cnt], 'threshold': [0], 'check': [cnt > 0]})

    @quality_check_decorator
    def check_isolation_forest(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        df.rename(columns={str(cfg['value_column']): 'value', str(cfg['date_column']): 'date'}, inplace=True)
        model = IsolationForest(contamination=float(self.threshold), random_state=42)
        df['anomaly_score'] = model.fit_predict(df[['value']].values)
        last_row = df.sort_values('date').iloc[-1]
        return pd.DataFrame({'date': [last_row['date']], 'check_value': [float(last_row['anomaly_score'])], 'threshold': [self.threshold], 'check': [last_row['anomaly_score'] == 1]})

    @quality_check_decorator
    def check_psi_stability(self):
        def calculate_psi(expected, actual, buckets=10):
            breakpoints = np.percentile(expected, np.arange(0, buckets + 1) / buckets * 100)
            breakpoints[0], breakpoints[-1] = -np.inf, np.inf
            exp_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            act_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
            exp_percents, act_percents = np.clip(exp_percents, 0.0001, 1), np.clip(act_percents, 0.0001, 1)
            return np.sum((exp_percents - act_percents) * np.log(exp_percents / act_percents))

        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        df.rename(columns={str(cfg['value_column']): 'value', str(cfg['date_column']): 'date'}, inplace=True)
        df = df.sort_values('date')
        psi_score = calculate_psi(df.iloc[:-1]['value'], df.iloc[-1:]['value'])
        return pd.DataFrame({'date': [df['date'].iloc[-1]], 'check_value': [round(psi_score, 4)], 'threshold': [self.threshold], 'check': [psi_score < float(self.threshold)]})

    def run(self):
        if self.skip: return None
        method = getattr(self, f"check_{self.check_type}", None)
        if method:
            summary, _ = method()
            return summary
        return None
