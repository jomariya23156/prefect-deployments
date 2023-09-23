import os
import json
import yaml
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from typing import Union, List, Dict
from collections import defaultdict
from evidently import ColumnMapping
from datetime import datetime, timedelta

CENTRAL_STORAGE_PATH = os.getenv('CENTRAL_STORAGE_PATH', '/service/central_storage')

# the best practice is to retrieve the model & config from a model registry service or cloud storage
# and this function should implement the logic to download files to local storage and read them
def retrieve_metadata_file(model_metadata_file_path: str) -> Dict:
    model_meta_path = os.path.join(CENTRAL_STORAGE_PATH, 'models', model_metadata_file_path)
    with open(model_meta_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata

def retrieve_ref_data_df(model_name: str) -> pd.DataFrame:
    ref_data_path = os.path.join(CENTRAL_STORAGE_PATH, 'ref_data', f'{model_name}_ref_data.parquet')
    ref_data_df = pd.read_parquet(ref_data_path)
    return ref_data_df

def open_db_session(engine: sqlalchemy.engine) -> sqlalchemy.orm.Session:
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def get_cur_df_from_query(sql_ret, use_cols: List[str] = ['id', 'uae_feats', 'bbsd_feats', 'prediction_json']) -> pd.DataFrame:
    current_data = defaultdict(list)
    for row in sql_ret:
        for col in use_cols:
            current_data[col].append(getattr(row, col))
    cur_df = pd.DataFrame(current_data).set_index('id')
    return cur_df

def convert_pred_json_to_df(pred_json_col: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(list(pred_json_col.apply(lambda x: json.loads(x))))

def create_col_mapping(classes, num_feat_cols, uae_feat_cols, bbsd_feat_cols) -> ColumnMapping:
    column_mapping = ColumnMapping()

    column_mapping.target = 'label'
    column_mapping.numerical_features = num_feat_cols
    column_mapping.prediction = classes
    column_mapping.embeddings = {'uae': uae_feat_cols, 'bbsd': bbsd_feat_cols}
    column_mapping.id = None
    column_mapping.datetime = None
    
    return column_mapping

def query_last_rows(session, table, model_name, last_days, last_n):
    q = session.query(table).filter(table.model_name == model_name)
    if last_days:
        days_ago = datetime.utcnow() - timedelta(days=last_days)
        days_ago_str = days_ago.strftime('%Y-%m-%d %H:%M:%S')
        # Query the rows added in the last 7 days regardless of database time zone
        q = q.filter(
                func.timezone('UTC', table.created_on) >= days_ago_str
            )
        if last_n:
            q = q.limit(last_n)
        ret = q.all()
    elif last_n:
        ret = q.order_by(table.created_on.desc()).limit(last_n).all()
    else:
        ret = q.order_by(table.created_on.desc()).all()
    return ret