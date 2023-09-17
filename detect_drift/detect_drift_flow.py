import os
import json 
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from collections import defaultdict

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from prefect import task, flow, get_run_logger

from evidently.ui.remote import RemoteWorkspace

from .evidently_tasks import (create_report, create_test_suite, modify_dashboard, 
                              make_cur_evidently_compat, make_ref_evidently_compat)
from .utils import (open_db_session, get_cur_df_from_query, convert_pred_json_to_df, 
                    create_col_mapping, retrieve_metadata_file, query_last_rows, retrieve_ref_data_df)

POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
EVIDENTLY_PORT = os.getenv('EVIDENTLY_PORT', '8080')
EVIDENTLY_URL = os.getenv('EVIDENTLY_URL', f'http://evidently:{EVIDENTLY_PORT}')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://dlservice_user:SuperSecurePwdHere@postgres:{POSTGRES_PORT}/dlservice_pg_db')
DB_PREDICTION_TABLE_NAME = os.getenv('DB_PREDICTION_TABLE_NAME', 'predictions')
CENTRAL_STORAGE_PATH = os.getenv('CENTRAL_STORAGE_PATH', '/service/central_storage')

# might have to create Prefect variables to store model_metadata.yml path
# then reference those variables in prefect.yaml and pass in as parameters to this flow function
@flow(name='detect_drift_with_evidently')
def detect_drift_flow(model_metadata_file_path: str, last_days: Optional[int]=7, last_n: Optional[int]=500,
                 evidently_project_name: Optional[str]='production_model_monitor',
                 evidently_project_desc: Optional[str]='Dashboard for monitoring production models'):
    logger = get_run_logger()
    if not model_metadata_file_path.endswith(('.yaml', '.yml')):
        raise ValueError("Invalid format. Parameter model_metadata_file_path does not end with .yaml or .yml")
    logger.info(f"Loading the model metadata from {os.path.join(CENTRAL_STORAGE_PATH, 'models', model_metadata_file_path)}")
    model_metadata = retrieve_metadata_file(model_metadata_file_path)
    classes = model_metadata['classes']
    engine = create_engine(DB_CONNECTION_URL)
    Base = automap_base()
    Base.prepare(autoload_with=engine)
    prediction_table_base = getattr(Base.classes, DB_PREDICTION_TABLE_NAME)
    
    session = open_db_session(engine)
    # latest N elements
    if (not last_days) and (not last_n):
        logger.warning('Both last_days & last_n are set to 0 or None, this will retrieve all rows from the table '+\
                       'and can take a long time to compute reports and test suites.')
    ret = query_last_rows(session, prediction_table_base, last_days, last_n)
    
    temp_cur_df = get_cur_df_from_query(ret, use_cols=['id', 'uae_feats', 'bbsd_feats', 'prediction_json'])
    cur_df, cur_num_feat_cols, cur_uae_feat_cols, cur_bbsd_feat_cols = make_cur_evidently_compat(temp_cur_df)

    temp_ref_df = retrieve_ref_data_df(model_name=model_metadata['model_name'])
    ref_df, ref_num_feat_cols, ref_uae_feat_cols, ref_bbsd_feat_cols = make_ref_evidently_compat(temp_ref_df, classes)
    
    if set(ref_df.columns).difference(set(cur_df.columns)) != set():
        raise Exception('Columns of ref and cur data are not equal, please reverify.')
    column_mapping = create_col_mapping(classes, ref_num_feat_cols, ref_uae_feat_cols, ref_bbsd_feat_cols)
    
    ws = RemoteWorkspace(EVIDENTLY_URL)
    search_results = ws.search_project(evidently_project_name)
    
    if len(search_results) == 0:
        logger.info('Created a new Evidently project')
        project = ws.create_project(evidently_project_name)
        project = modify_dashboard(project, project_desc=evidently_project_desc)
    else:
        # select the latest one
        logger.info('Evidently project already exists. Use the latest one.')
        project = search_results[-1]
        
    report = create_report()
    test_suite = create_test_suite()
    logger.info('Computing Reports & Test suites')
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
    test_suite.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
    logger.info('Adding Reports & Test suites to the project')
    ws.add_report(project.id, report)
    ws.add_test_suite(project.id, test_suite)
    logger.info(f'Report id: {report.id}')
    logger.info(f'Test suite id: {test_suite.id}')
    logger.info('Successfully added Reports & Test suites to the project')