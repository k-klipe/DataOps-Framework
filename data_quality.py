import os
import pandas as pd
import numpy as np
from functools import wraps
import yaml
from modules.logging import get_logging
import json
import requests
from datetime import datetime
from sklearn.ensemble import IsolationForest
from prophet import Prophet

class DataQualityCheck:
    DEFAULT_MATTERMOST_CONN = 'mm'

    def init(self, project, parent_project, config, source_sql=None, target_sql=None):
        self.config = config
        self.project = project
        self.parent_project = parent_project
        self.check_name = self.config.get('check_name')
        self.owner = self.config.get('owner', '')
        self.alert_users = self.config.get('alert_users', '')
        self.check_type = self.config.get('check_type')
        self.schedule = self.config.get('schedule')
        self.queries = self.config.get('queries', {})
        self.threshold = self.config.get('threshold', 0.1)
        self.abs = self.config.get('abs', False)
        self.larger = self.config.get('larger', False)
        self.text = self.config.get('text', '')
        self.send_always = self.config.get('send_always', False)
        self.send_graph = self.config.get('send_graph', False)
        self.send_details = self.config.get('send_details', False)
        self.skip = self.config.get('skip', False)
        self.priority = self.config.get('priority')
        
        self.source_sql = source_sql
        self.target_sql = target_sql

    def should_run(self, interval_start, current_time):
        from croniter import croniter
        if not croniter.is_valid(self.schedule):
            return 'wrong'
        if self.skip:
            return 'skip'
        
        cron = croniter(self.schedule, interval_start)
        run = cron.get_next(datetime)
        return 'run' if interval_start <= run <= current_time else 'skip'

    def setup_connection(self, db, connection_id):
        from airflow.hooks.base import BaseHook
        from clickhouse_driver import Client
        from sqlalchemy import create_engine
        from urllib.parse import quote_plus

        conn_dict = BaseHook.get_connection(connection_id)
        
        if db == 'vertica':
            vertica_prod = f'vertica+vertica_python://{conn_dict.login}:{quote_plus(conn_dict.password)}@{conn_dict.host}:{conn_dict.port}/{conn_dict.schema}'
            return create_engine(vertica_prod, pool_pre_ping=True)
        elif db == 'clickhouse':
            return Client(host=conn_dict.host, user=conn_dict.login, password=conn_dict.password, port=conn_dict.port)
        else:
            raise ValueError(f'Тип подключения {db} не поддерживается текущей версией.')

    def execute_query(self, db, connection_id, query, value_column='', date_column=''):
        from clickhouse_driver import Client
        engine = self.setup_connection(db, connection_id)
        
        if isinstance(engine, Client):
            result_data, columns = engine.execute(query, with_column_types=True)
            result_df = pd.DataFrame(result_data, columns=[col[0] for col in columns])
        else:
            result_df = pd.read_sql(query, engine)
            
        return result_df

    def send_errors_to_mattermost(self, mattermost_conn_id, errors_text):
        from airflow.hooks.base import BaseHook
        from mattermostdriver import Driver
        conn = BaseHook.get_connection(mattermost_conn_id)
        driver = Driver({'url': conn.host, 'token': conn.password, 'scheme': "https", 'port': 443, 'basepath': "/api/v4"})
        
        try:
            driver.login()
            msg = f"{self.alert_users}\nПроект: {self.parent_project or ''} {self.project}\nПроверка: {self.check_name}\nОшибка сборки данных"
            resp = driver.posts.create_post({'channel_id': conn.schema, 'message': msg})
            driver.posts.create_post({'channel_id': conn.schema, 'message': errors_text, 'root_id': resp['id']})
        except Exception as e:
            logging.error(f"Mattermost error: {e}")

    def quality_check(check_function):
        @wraps(check_function)
        def wrapper(self, *args, **kwargs):
            logging.info(f'Running {check_function.__name__}')
            full_df = check_function(self, *args, **kwargs)
            if self.check_type == 'table_empty':
                return full_df, full_df
            return full_df[['date', 'check_value', 'threshold', 'check']], full_df
        return wrapper

    @quality_check
    def check_percentage_deviation(self):
        sources = ['source', 'target']
        dfs = {}
        for s in sources:
            cfg = self.queries[s]
            sql = self.source_sql if s == 'source' else self.target_sql
            df = self.execute_query(cfg['db'], cfg['connection_id'], sql, cfg.get('value_column'), cfg.get('date_column'))
            
            if df.empty:
                self.send_errors_to_mattermost(self.DEFAULT_MATTERMOST_CONN, f"Пустой результат в {s}: {sql}")
            
            df.rename(columns={str(cfg['value_column']): f'{s}_value', str(cfg['date_column']): 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            dfs[s] = df

        check_df = pd.merge(dfs['source'], dfs['target'], on='date', how='outer').dropna(subset=['source_value', 'target_value'])
        
        def calc_diff(row):
            if row['target_value'] == 0: return None
            diff = row['source_value'] - row['target_value']
            val = abs(diff) / row['target_value'] if self.abs else diff / row['target_value']
            return round(val, 3)

        check_df['check_value'] = check_df.apply(calc_diff, axis=1)
        check_df['threshold'] = self.threshold
        check_df['check'] = check_df['check_value'].apply(lambda x: x >= self.threshold if self.larger else x <= self.threshold)
        return check_df

    @quality_check
    def check_standart_deviation(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql, cfg.get('value_column'), cfg.get('date_column'))
        df.rename(columns={str(cfg['value_column']): 'source_value', str(cfg['date_column']): 'date'}, inplace=True)
        df = df.sort_values(by='date', ascending=False)
        
        date, last_val = df['date'].iloc[0], df['source_value'].iloc[0]
        hist = df['source_value'].iloc[1:]
        mean_v, std_v = hist.mean(), round(hist.std(), 6)
        
        dev = round(abs((last_val - mean_v) / std_v) if std_v != 0 else abs(last_val - mean_v), 3)
        return pd.DataFrame({'date': [date], 'check_value': [dev], 'threshold': [self.threshold], 'check': [dev <= self.threshold]})

    @quality_check
    def check_compare_threshold(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql, cfg.get('value_column'), cfg.get('date_column'))
        df.rename(columns={str(cfg['value_column']): 'source_value', str(cfg['date_column']): 'date'}, inplace=True)
        
        df['check_value'] = df['source_value'].abs().round(3) if self.abs else df['source_value'].round(3)
        df['threshold'] = self.threshold
        df['check'] = df['check_value'].apply(lambda x: x >= self.threshold if self.larger else x <= self.threshold)
        return df

    @quality_check
    def check_table_empty(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        cnt = df.iloc[0, 0]
        return pd.DataFrame({'date': [datetime.now()], 'check_value': [cnt], 'threshold': [None], 'check': [cnt > 0]})

    @quality_check
    def check_isolation_forest(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        df.rename(columns={str(cfg['value_column']): 'value', str(cfg['date_column']): 'date'}, inplace=True)
        data = df[['value']].values
        model = IsolationForest(contamination=float(self.threshold), random_state=42)
        df['anomaly_score'] = model.fit_predict(data)
        last_row = df.sort_values('date').iloc[-1]
        is_ok = last_row['anomaly_score'] == 1
        return pd.DataFrame({'date': [last_row['date']], 'check_value': [float(last_row['anomaly_score'])], 'threshold': [self.threshold], 'check': [is_ok]})

    @quality_check
    def check_prophet_forecasting(self):
        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        df.rename(columns={str(cfg['value_column']): 'y', str(cfg['date_column']): 'ds'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        train_df = df.sort_values('ds').iloc[:-1]
        actual_row = df.sort_values('ds').iloc[-1]
        model = Prophet(interval_width=0.95)
        model.fit(train_df)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        res = forecast.iloc[-1]
        fact = actual_row['y']
        is_ok = res['yhat_lower'] <= fact <= res['yhat_upper']
        return pd.DataFrame({'date': [actual_row['ds']], 'check_value': [round(float(fact), 3)], 'threshold': [f"{round(res['yhat_lower'],1)}-{round(res['yhat_upper'],1)}"], 'check': [is_ok]})

    @quality_check
    def check_psi_stability(self):
        def calculate_psi(expected, actual, buckets=10):
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            breakpoints = np.percentile(expected, breakpoints)
            breakpoints[0], breakpoints[-1] = -np.inf, np.inf
            expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
            expected_percents, actual_percents = np.clip(expected_percents, 0.0001, 1), np.clip(actual_percents, 0.0001, 1)
            return np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

        cfg = self.queries['source']
        df = self.execute_query(cfg['db'], cfg['connection_id'], self.source_sql)
        df.rename(columns={str(cfg['value_column']): 'value', str(cfg['date_column']): 'date'}, inplace=True)
        df = df.sort_values('date')
        psi_score = calculate_psi(df.iloc[:-1]['value'], df.iloc[-1:]['value'])
        is_ok = psi_score < float(self.threshold)
        return pd.DataFrame({'date': [df['date'].iloc[-1]], 'check_value': [round(psi_score, 4)], 'threshold': [self.threshold], 'check': [is_ok]})

    def send_results_to_mattermost(self, mattermost_conn_id, check_df, is_percent=False, full_df=None):
        # Логика отправки алертов
        pass 

    def run_check(self):
        check_type = self.config['check_type']
        check_method = getattr(self, f"check_{check_type}")
        check_df, full_df = check_method()
        self.send_results_to_mattermost(self.DEFAULT_MATTERMOST_CONN, check_df, full_df=full_df)
