import os
import pandas as pd
import numpy as np


def process_data(recommendation_dir, path_experiment_data, path_user_job_split, run, k):
    if not os.path.exists(os.path.join(recommendation_dir, "top_" + str(k), str(run))):
        os.makedirs(os.path.join(recommendation_dir, "top_" + str(k), str(run)))

    path_user = "./DATA/users.csv"
    data_users = pd.read_csv(path_user)
    data_users['country'] = data_users['country'].apply(lambda x: 'non-de' if x != 'de' else x)

    path_items = "./DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items['country'] = data_items['country'].apply(lambda x: 'non-de' if x != 'de' else x)


    file_path = os.path.join(path_experiment_data, "out-" + str(run) + ".txt")
    
    recommendation = read_file(file_path, sep=',', columns=['U_ID', 'J_ID', 'S']).groupby('U_ID').apply(
        lambda x: x[:k]).reset_index(drop=True)
    recommendation = add_rank_column(recommendation, 'U_ID')
    recommendation = add_sensitive_columns(recommendation, 'J_ID', data_items,
                                                    ['country', 'is_payed'])
    recommendation = add_sensitive_columns(recommendation, 'U_ID', data_users,
                                                    ['country', 'premium'])

    if not os.path.exists(os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_test_user_job.csv")):
        file_path = os.path.join(path_user_job_split, "test_interactions_userjob_binary.txt")
        data_test_user_job = read_file(file_path, sep='\t', columns=['U_ID', 'J_ID', 'interaction'])
        data_test_user_job = add_sensitive_columns(data_test_user_job, 'J_ID', data_items, ['country', 'is_payed'])
        data_test_user_job = add_sensitive_columns(data_test_user_job, 'U_ID', data_users, ['country', 'premium'])
        data_test_user_job.to_csv(
            os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_test_user_job.csv"))
    else:
        data_test_user_job = pd.read_csv(
            os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_test_user_job.csv"))

    if not os.path.exists(os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_train_user_job.csv")):
        file_path = os.path.join(path_user_job_split, "train_interactions_userjob_binary.txt")
        data_train_user_job = read_file(file_path, sep='\t', columns=['U_ID', 'J_ID', 'interaction'])
        data_train_user_job = add_sensitive_columns(data_train_user_job, 'J_ID', data_items, ['country', 'is_payed'])
        data_train_user_job = add_sensitive_columns(data_train_user_job, 'U_ID', data_users, ['country', 'premium'])
        data_train_user_job.to_csv(
        os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_train_user_job.csv"))
    else:
        data_train_user_job = pd.read_csv(os.path.join(recommendation_dir, "top_" + str(k), str(run), str(run) + "_data_train_user_job.csv"))
    return recommendation, data_train_user_job, data_test_user_job, data_items, data_users

def add_sensitive_columns(data, id_column, data_sensitive, columns_sensitive):
    mask = data[id_column].isin(data_sensitive["id"])
    data = data[data[id_column].isin(data_sensitive["id"])]
    for s in columns_sensitive:
        if s not in data:
            col_name = s
        else:
            col_name = s + "__" + id_column
        data[col_name] = data[id_column].map(data_sensitive.set_index('id')[s])
    return data

def add_intersectional_group_column(data, columns_sensitive):
    column_name = ""
    for s in columns_sensitive:
        column_name = column_name + "__" + s
    data[column_name] = [""] * len(data)
    for s in columns_sensitive:
        data[column_name] = data[column_name] + "__" + data[s].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))

    return data

def read_file(file_path, sep, columns):
    data = pd.read_csv(file_path, sep=sep, header=None)
    data.columns = columns

    for col in columns:
        data[col] = data[col].apply(lambda x: float(x))
    return data


def add_rank_column(data, query_col):
    data['rank'] = data.groupby(query_col).cumcount()
    data['rank'] = data['rank'] + 1

    return data


