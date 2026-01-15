import itertools
import os
import time
import numpy as np
import pandas as pd
import argparse

from src.utils import process_data, add_intersectional_group_column
from src.evaluation.fairness_metrics import compute_metrics
from src.evaluation.compatibility_score import compatibility_score, compatibility_analysis
from src.configs import get_configs

def run_fairness_metrics(recommendation, data_test_user_job, data_train_user_job, data_items, save_path, k):
    recommendation = add_intersectional_group_column(recommendation, ['country', 'is_payed'])
    data_test_user_job = add_intersectional_group_column(data_test_user_job, ['country', 'is_payed'])
    data_train_user_job = add_intersectional_group_column(data_train_user_job, ['country', 'is_payed'])
    data_items = add_intersectional_group_column(data_items, ['country', 'is_payed'])

    data_test_user_job = add_intersectional_group_column(data_test_user_job, ['country__U_ID', 'premium'])
    data_train_user_job = add_intersectional_group_column(data_train_user_job, ['country__U_ID', 'premium'])

    data_dict_user_job = {
            "query_col": 'U_ID',
            "groups": ['country', 'is_payed', '__country__is_payed'],  
            "recommendation": recommendation,
            "proportional_item_population": data_items,
            "proportional_user_population": data_test_user_job,
            "proportional_user_preference": data_train_user_job,
        }
    

    # fairness non-proportional or proportional to the group population of the rec items
    # e.g. items are the jobs and in the whole data we have 10% non-de jobs and 90% de jobs,
    # we expect the rec to be proportional to this for all users (queries)
    compute_metrics(data_dict_user_job, k, save_path)

   
    # fairness non-proportional or proportional to the group population of the rec items (global means that we don't compute this per indiviudal user, but overall)
    # e.g. items are the jobs and in the whole data we have 10% non-de jobs and 90% de jobs,
    # we expect the rec to be proportional to this for all users (queries)
    compute_metrics(data_dict_user_job, k, save_path, global_population=False)
    compute_metrics(data_dict_user_job, k, save_path, global_population=True)

    # fairness proportional to the user's (query's) preferences (clicks on the rec items)
    # e.g. items are the jobs which are rec to users. if user A clicked on 10% of the time on non-de jobs
    # and in 90% of the time on de jobs, we expect the rec for this user to be proportional to this distribution
    compute_metrics(data_dict_user_job, k, save_path, per_ID_prop=True)

def run_compatibility_score(recommendation, data_users, data_items, save_path, fields=['industry_id', 'discipline_id', 'career_level', 'country']):
    # run compatibility score
    print("Computing compatibility score...")
    save_dir = os.path.join(save_path, "compatibility")
    os.makedirs(save_dir, exist_ok=True)
    file_name="compatibility_user_job.csv"
    compatibility_score(
        recommendation,
        data_users,
        data_items,
        save_dir,
        file_name=file_name,
        fields=fields
    )

    # run compatibility analysis of added vs removed in comparison to 1-sided rec (BPR)
    dict_experiment_data_1sided = get_configs(method="UserJob", core="20", k=10, args_bool=False)
    recommendation_1sided, _, _, data_items,  data_users = process_data(dict_experiment_data_1sided["recommendation_dir"], dict_experiment_data_1sided["data_path"],
    dict_experiment_data_1sided["path_user_job_split"], 1, dict_experiment_data_1sided["k"])

    print("Computing compatibility analysis...")
    compatibility_analysis(
        recommendation_1sided,
        recommendation,
        data_users,
        data_items,
        "country",
        "de",
        "non-de",
        save_path,
        fields=fields
    )
    compatibility_analysis(
        recommendation_1sided,
        recommendation,
        data_users,
        data_items,
        "is_payed",
        0,
        1,
        save_path,
        fields=fields
    )

def run(dict_experiment_data):
    path_experiment_data = dict_experiment_data["data_path"]
    path_user_job_split = dict_experiment_data["path_user_job_split"]
    recommendation_dir = dict_experiment_data["recommendation_dir"]
    out_path =dict_experiment_data["out_path"]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for run in range(1, 2):
        start_time = time.time()
        recommendation, data_train_user_job, data_test_user_job, data_items, data_users = process_data(
            recommendation_dir,
            path_experiment_data,
            path_user_job_split, run, dict_experiment_data["k"])
        
        end_time = time.time()
        print("Elapsed time process data", end_time - start_time)

        save_path = dict_experiment_data["save_path"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        run_fairness_metrics(recommendation, data_test_user_job, data_train_user_job, data_items, save_path, dict_experiment_data["k"])
        fields = ['industry_id', 'discipline_id', 'career_level']
        run_compatibility_score(recommendation, data_users, data_items, dict_experiment_data["out_path"], fields)

if __name__ == "__main__":
    dict_experiment_data = get_configs()
    run(dict_experiment_data)
