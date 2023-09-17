import numpy as np
import pandas as pd
from prefect import task
from typing import Union, List, Dict

from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace, Project
from evidently.metrics import ConflictPredictionMetric, EmbeddingsDriftMetric, ConflictTargetMetric
from evidently.tests import TestConflictPrediction, TestEmbeddingsDrift, TestConflictTarget
from evidently.ui.dashboards import CounterAgg, DashboardPanelCounter, PanelValue, ReportFilter
from evidently.metrics.data_drift.embedding_drift_methods import model, ratio, distance, mmd

from .utils import convert_pred_json_to_df

@task(name='create_report')
def create_report() -> Report:
    data_drift_report = Report(
        metrics=[
            ConflictTargetMetric(),
            ConflictPredictionMetric(),
            # mmd with uae # note: currently, mmd failed. please check docs
            EmbeddingsDriftMetric('uae', drift_method = mmd(threshold = 0.015, quantile_probability = 0.95)),
            # ks with uae
            EmbeddingsDriftMetric('uae', drift_method = ratio(
                                        component_stattest='ks',
                                        threshold = 0.05
                                    )
                                 ),
            # euclidean with uae
            EmbeddingsDriftMetric('uae', 
                                  drift_method = distance(
                                      dist = 'euclidean',
                                      threshold = 0.2
                                  )
                                 ),
            # model with uae
            EmbeddingsDriftMetric('uae', drift_method = model(threshold = 0.75)),
            # mmd with bbsd
            EmbeddingsDriftMetric('bbsd', drift_method = mmd(threshold = 0.015, quantile_probability = 0.95)),
            # ks with bbsd
            EmbeddingsDriftMetric('bbsd', drift_method = ratio(
                                        component_stattest='ks',
                                        threshold = 0.05
                                    )
                                 ),
        ]
    )
    return data_drift_report

@task(name='create_test_suite')
def create_test_suite() -> TestSuite:
    data_drift_test_suite = TestSuite(
        tests=[
            TestConflictTarget(),
            TestConflictPrediction(),
            # mmd with uae
            TestEmbeddingsDrift('uae', drift_method = mmd(threshold = 0.015, quantile_probability = 0.95)),
            # ks with uae
            TestEmbeddingsDrift('uae', drift_method = ratio(
                                        component_stattest='ks',
                                        threshold = 0.05
                                    )
                                 ),
            # euclidean with uae
            TestEmbeddingsDrift('uae', 
                                  drift_method = distance(
                                      dist = 'euclidean',
                                      threshold = 0.2
                                  )
                                 ),
            # model with uae
            TestEmbeddingsDrift('uae', drift_method = model(threshold = 0.75)),
            # mmd with bbsd
            TestEmbeddingsDrift('bbsd', drift_method = mmd(threshold = 0.015, quantile_probability = 0.95)),
            # ks with bbsd
            TestEmbeddingsDrift('bbsd', drift_method = ratio(
                                        component_stattest='ks',
                                        threshold = 0.05
                                    )
                                 ),
        ]
    )
    return data_drift_test_suite

@task(name='modify_dashboard')
def modify_dashboard(project, project_desc: str) -> Project:
    project.description = project_desc
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Production Model Monitor!",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Production Data Quality",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Number of conflicts in Prediction",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ConflictPredictionMetric",
                field_path=ConflictPredictionMetric.fields.current.number_not_stable_prediction,
            ),
            text="count",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Number of conflicts in Target (GT)",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ConflictTargetMetric",
                field_path=ConflictTargetMetric.fields.number_not_stable_target,
            ),
            text="count",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Drift Detection",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Drift Score",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="EmbeddingsDriftMetric",
                field_path=EmbeddingsDriftMetric.fields.drift_score,
                legend="score",
            ),
            text="latest",
            agg=CounterAgg.LAST,
            size=2,
        )
    )
    project.save()
    return project

@task(name='make_current_data_format_evidently_compatible')
def make_cur_evidently_compat(cur_df: pd.DataFrame, uae_feats_col: str= 'uae_feats',
                              bbsd_feats_col: str='bbsd_feats', pred_json_col: str = 'prediction_json'):
    uae_feats_arr = np.stack(cur_df[uae_feats_col])
    uae_n_feats = uae_feats_arr.shape[1]
    uae_feat_cols = [f'uae_feat_{i}' for i in range(uae_n_feats)]
    uae_df = pd.DataFrame(uae_feats_arr, columns=uae_feat_cols)
    # Create a dup of uae_feat to use for a different purpose
    # uae_feat will be used as embeddings for computing drift
    # numerical_feat will be used as numerical features for the data quality test
    # specifically, TestConflictTarget & TestConflictPrediction
    # which are useful to verify. Evidently does not support using the same columns
    # twice in column mapping. So, we have to create a duplicate and save them alongside here
    num_feat_cols = [f'numerical_feat_{i}' for i in range(uae_n_feats)]
    num_df = pd.DataFrame(uae_feats_arr, columns=num_feat_cols)

    bbsd_feats_arr = np.stack(cur_df[bbsd_feats_col])
    bbsd_n_feats = bbsd_feats_arr.shape[1]
    bbsd_feat_cols = [f'bbsd_feat_{i}' for i in range(bbsd_n_feats)]
    bbsd_df = pd.DataFrame(bbsd_feats_arr, columns=bbsd_feat_cols)

    pred_df = convert_pred_json_to_df(cur_df[pred_json_col])
    final_df = pd.concat([num_df, uae_df, bbsd_df, pred_df], axis=1)

    # fill columns that exist in ref but not in this cur (label col) with nan
    # to make schema of both ref and cur df identical
    final_df['label'] = [np.nan] * len(final_df)
    
    return final_df, num_feat_cols, uae_feat_cols, bbsd_feat_cols

@task(name='make_reference_data_format_evidently_compatible')
def make_ref_evidently_compat(ref_df: pd.DataFrame, classes: List[str], uae_feats_col: str= 'uae_feats',
                              bbsd_feats_col: str='bbsd_feats', label_col: str = 'label'):
    uae_feats_arr = np.stack(ref_df[uae_feats_col])
    uae_n_feats = uae_feats_arr.shape[1]
    uae_feat_cols = [f'uae_feat_{i}' for i in range(uae_n_feats)]
    uae_df = pd.DataFrame(uae_feats_arr, columns=uae_feat_cols)
    # Create a dup of uae_feat to use for a different purpose
    num_feat_cols = [f'numerical_feat_{i}' for i in range(uae_n_feats)]
    num_df = pd.DataFrame(uae_feats_arr, columns=num_feat_cols)
    
    bbsd_feats_arr = np.stack(ref_df[bbsd_feats_col])
    bbsd_n_feats = bbsd_feats_arr.shape[1]
    bbsd_feat_cols = [f'bbsd_feat_{i}' for i in range(bbsd_n_feats)]
    bbsd_df = pd.DataFrame(bbsd_feats_arr, columns=bbsd_feat_cols)
    
    final_df = pd.concat([num_df, uae_df, bbsd_df], axis=1)
    

    # fill columns that exist in cur but not in this ref (prediction cols) with nan
    # to make schema of both ref and cur df identical
    for class_name in classes:
        final_df[class_name] = [np.nan] * len(final_df)

    final_df['label'] = ref_df[label_col].apply(lambda x: np.argmax(x))
    
    return final_df, num_feat_cols, uae_feat_cols, bbsd_feat_cols