from src.evaluation.fairness_metrics import compute_metric
from src.utils import add_sensitive_columns, add_rank_column
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import os
import time
import torch


def fair_fusion_optimisation(user_job, job_user, core, k, fairness_metrics, group_cols, w_uf=None):
    # learn a w for each U_ID such that we minimize the compund metric (global fairness + preference)

    print("Setup...")
    print(fairness_metrics)
    print(group_cols)
    print("Get constants...")
    fairness_metric_dict = constants(group_cols, core, k)

    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)

    # create input vecotr X [uid, Su, Sj, group1, group 2, ...]
    uids = merged['U_ID'].values      
    jids = merged['J_ID'].values      
    Su = merged['score_user_job'].values      
    Sj = merged['score_job_user'].values
    group_values = []
    for group_col in group_cols:
        group_values.append(merged[group_col + '_user_job'].values)  
    A = np.column_stack(group_values)
    X = np.column_stack((uids, jids, Su, Sj, A)) 

    w_per_uid = {uid: 0.5 for uid in set(uids)}

    def score(w):
        w_temp = w_per_uid.copy()
        w_temp[uid] = w

        if "DGI(gloabl)" in fairness_metrics or "EGI(global)" in fairness_metrics:
            return compound_metric(X, w_temp, fairness_metrics, group_cols, fairness_metric_dict, w_uf)
        else:
            return compound_metric(X_uid, w_temp, fairness_metrics, group_cols, fairness_metric_dict, w_uf)

    for uid in set(uids):
        start_time = time.time()
        if not ("DGI(gloabl)" in fairness_metrics or "EGI(global)" in fairness_metrics):
            X_uid = X[X[:, 0] == uid]
            
        log_file = "logs_fair_fusion_optim.txt"
        best_w = min(np.linspace(0, 1, 101), key=score)
        w_per_uid[uid] = best_w
        end_time = time.time()

        with open(os.path.join("./", log_file), 'a') as f:
                f.write(str(fairness_metrics))
                f.write(str(group_cols))
                f.write(f"time uid {uid}: {end_time - start_time:.4f} seconds" + "\n")

    user_ids = user_job["U_ID"].unique()
    uid_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    user_job["w"] = user_job["U_ID"].map(lambda uid: w_per_uid[uid])
    job_user["w"] = job_user["U_ID"].map(lambda uid: w_per_uid[uid])

    return compute_weighted_sum_fusion(user_job, job_user)


def compound_metric(X, w_per_uid, fairness_metrics, group_cols, fairness_metric_dict, w_uf=None):
    X_fused = fusion(X, w_per_uid)
    X_fused_top = top_k(X_fused)

    compund_metric_value = 0
    for index, fairness_metric in enumerate(fairness_metrics):
        for index_group, group_col in enumerate(group_cols):
            fairness_settings = fairness_metric_dict[fairness_metric]
            metric_value = compute_fairness_metric(X_fused_top, index_group + 3, group_col, fairness_settings)
            
            if w_uf is not None:
                if index == 0:
                    metric_value = w_uf * metric_value
                else:
                    metric_value = (1 - w_uf) * metric_value
                
            compund_metric_value += metric_value

    return compund_metric_value


def compute_weighted_sum_fusion(user_job, job_user):
    merged = pd.merge(user_job, job_user, on=['U_ID', 'J_ID', 'w'], suffixes=('_user_job', '_job_user'), how='outer')
    merged[['score_user_job', 'score_job_user']] = merged[['score_user_job', 'score_job_user']].fillna(0)
    merged['score'] = merged['w'] * merged['score_user_job'] + (1 - merged['w']) * merged['score_job_user']

    return merged[['U_ID', 'J_ID', 'score', 'w']]

def fusion(X, w_per_uid):
    # Map each UID to its w using vectorized lookup
    uids = X[:, 0]
    jids = X[:, 1]
    Su = X[:, 2].astype(float)
    Sj = X[:, 3].astype(float)

    # Map each UID to its w using vectorized lookup
    w_vec = np.vectorize(w_per_uid.get)(uids).astype(float)

    # Compute fusion scores
    fusion_scores = w_vec * Su + (1 - w_vec) * Sj

    # Concatenate back into final array
    rest = X[:, 4:]  # group columns
    fused_X = np.column_stack((uids, jids, fusion_scores, rest))
    return fused_X


def top_k(X, k=10):
    uids = X[:, 0]
    jids = X[:, 1]
    scores = X[:, 2].astype(float)
    
    # Unique UIDs and inverse mapping
    _, inverse_indices = np.unique(uids, return_inverse=True)
    num_users = inverse_indices.max() + 1

    # Estimate total size: num_users × k
    est_total = num_users * k
    n_features = X.shape[1]
    output = np.empty((est_total, n_features + 1), dtype=object)  # +1 for rank
    count = 0

    for user_id in range(num_users):
        idx = np.flatnonzero(inverse_indices == user_id)
        if idx.size == 0:
            continue

        user_scores = scores[idx]

        if idx.size <= k:
            topk_idx = idx
            ranks = np.argsort(-user_scores) + 1  # Rank starts from 1
        else:
            # Partial sort to get top-k, then sort for rank
            topk_local = np.argpartition(-user_scores, k - 1)[:k]
            topk_scores = user_scores[topk_local]
            sorted_local = topk_local[np.argsort(-topk_scores)]
            topk_idx = idx[sorted_local]
            ranks = np.arange(1, len(topk_idx) + 1)  # Start at 1

        for i, row_idx in enumerate(topk_idx):
            output[count] = np.append(X[row_idx], ranks[i])
            count += 1

    return output[:count]
   

def compute_fairness_metric(X, index_group, group_col, fairness_settings, k=10):
    uids = np.unique(X[:, 0])

    total_gap = 0
    count = 0

    if fairness_settings["global_population"]:
        count1_all = 0
        count2_all = 0

    for uid in set(uids):
        uid_mask = X[:, 0] == uid
        user_X = X[uid_mask]

        if fairness_settings["per_ID_prop"]:
            if int(uid) not in fairness_settings["data_proportional"]:
                print(f'{uid} not in fairness_settings - data_proportional')
                continue  # skip if no proportions available
        
        if fairness_settings["per_ID_prop"]:
            value_groups = list(fairness_settings["data_proportional"][uid][group_col].keys())
        else:
            value_groups = list(fairness_settings["data_proportional"][group_col].keys())
        if len(value_groups) != 2:
            continue


        group1, group2 = value_groups

        if fairness_settings["metric"] == "diversity":
            
            count1 = group_attribute_count(user_X, index_group, group1)
            count2 = group_attribute_count(user_X, index_group, group2)
        else:
            count1 = group_attribute_exposure(user_X, index_group, group1)
            count2 = group_attribute_exposure(user_X, index_group, group2)

        if fairness_settings["per_ID_prop"]:
            prop1 = fairness_settings["data_proportional"][uid][group_col][group1]
            prop2 = fairness_settings["data_proportional"][uid][group_col][group2]
        else:
            prop1 = fairness_settings["data_proportional"][group_col][group1]
            prop2 = fairness_settings["data_proportional"][group_col][group2]

        if not fairness_settings["global_population"]:
            rate1 = compute_prop(count1, prop1)
            rate2 = compute_prop(count2, prop2)

            gap = abs(rate1 - rate2)
            total_gap += gap
        else:
            count1_all += count1
            count2_all += count2

        count += 1

    if fairness_settings["global_population"]:
        total_selected = count * k

        count1_all = count1_all/ total_selected
        count2_all = count2_all/ total_selected

        rate1 = compute_prop(count1_all, prop1)
        rate2 = compute_prop(count2_all, prop2)

        total_gap = abs(rate1 - rate2)
        return total_gap 

    return total_gap / count if count > 0 else 0


def group_attribute_count(X, index_group, value_group):
    return np.sum(X[:, index_group] == value_group) / len(X)

def group_attribute_exposure(X, index_group, value_group):
    mask = X[:, index_group] == value_group
    if not np.any(mask):
        return 0.0

    index_rank = 4
    ranks = X[mask, index_rank].astype(float)
    exposure = np.sum(1 / np.log2(2 + ranks)) / len(X)
    return exposure


def compute_prop(count, prop):
    if count == 0 and prop == 0:
        return 1
    elif count == 0:
        return 0
    elif prop == 0:
        return count / 0.01
    else:
        return count / prop

def compute_data_proportional_all(data_items, group_cols, per_ID_prop=False):
    """
    Build nested dictionary for all group columns:
    data_proportional[uid][group_col][group_value] = count

    Parameters:
        data_items : pd.DataFrame with at least ['U_ID'] and group columns
        group_cols : list of str, columns to process (e.g. ['country', 'premium'])

    Returns:
        dict: {uid: {group_col: {group_value: count}}}
    """
    data_proportional = {}

    for group_col in group_cols:
        if group_col not in data_proportional:
            data_proportional[group_col] = {}
        if not per_ID_prop:
            grouped = data_items.groupby([group_col]).size()
            total_count = grouped.sum()
            for group_value, count in grouped.items():
                    if group_value not in data_proportional[group_col]:
                        data_proportional[group_col][group_value] = count / total_count
        else:
            grouped = data_items.groupby(['U_ID', group_col]).size()
            uid_totals = grouped.groupby(level=0).sum()

            for (uid, group_value), count in grouped.items():
                if uid not in data_proportional:
                    data_proportional[int(uid)] = {}
                if group_col not in data_proportional[int(uid)]:
                    data_proportional[int(uid)][group_col] = {}
                if group_value not in data_proportional[uid][group_col]:
                    data_proportional[int(uid)][group_col][group_value] = count / uid_totals[uid]

    return data_proportional

def constants(group_cols, core, k):
    recommendation_dir = f"./DATA/recommendation/core-{core}"

    data_test_user_job = pd.read_csv(
            os.path.join(recommendation_dir, "top_" + str(k), str(1), str(1) + "_data_test_user_job.csv"))
    data_train_user_job = pd.read_csv(
            os.path.join(recommendation_dir, "top_" + str(k), str(1), str(1) + "_data_train_user_job.csv"))
    
    path_items = "./DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items['country'] = data_items['country'].apply(lambda x: 'non-de' if x != 'de' else x)

    propoational_dict = {
        "proportional_item_population": data_items,
        "proportional_user_population": data_test_user_job,
        "proportional_user_preference": data_train_user_job,
    }

    propoational_dict = {
        "proportional_item_population_global": compute_data_proportional_all(data_items, group_cols),
        "proportional_item_population_UID": compute_data_proportional_all(data_items, group_cols),
        "proportional_user_preference": compute_data_proportional_all(data_train_user_job, group_cols, per_ID_prop=True),
    }


    fairness_metric_dict = {
        "DGI":{
            "metric": "diversity",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_item_population_UID"],
            "per_ID_prop": False,
            "global_population": False
        },
        "DGI(global)":{
            "metric": "diversity",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_item_population_global"],
            "per_ID_prop": False,
            "global_population": True
        },
        "DUP":{
            "metric": "diversity",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_user_preference"],
            "per_ID_prop": True,
            "global_population": False
        },
        "EGI":{
            "metric": "exposer",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_item_population_UID"],
            "per_ID_prop": False,
            "global_population": False
        },
        "EGI(global)":{
            "metric": "exposer",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_item_population_global"],
            "per_ID_prop": False,
            "global_population": True
        },
        "EUP":{
            "metric": "exposer",
            "group_user": None,
            "data_proportional": propoational_dict["proportional_user_preference"],
            "per_ID_prop": True,
            "global_population": False
        }
    }

    return fairness_metric_dict



   
