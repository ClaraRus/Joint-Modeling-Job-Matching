import os
import pandas as pd
import numpy as np
from src.utils import add_sensitive_columns, add_rank_column

def read_data(core, index):
    path_items = "./DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items['country'] = data_items['country'].apply(lambda x: 'non-de' if x != 'de' else x)

    recommendation_userjob, recommendation_jobuser = read_data_core_20(data_items, index)

    min_value = recommendation_userjob[recommendation_userjob["score"] != -np.inf]["score"].min() - 0.01
    recommendation_userjob.replace(-np.inf, min_value, inplace=True)

    recommendation_userjob["score"] = recommendation_userjob["score"] + (-1) * recommendation_userjob["score"].min() + 0.01

    min_value = recommendation_jobuser[recommendation_jobuser["score"] != -np.inf]["score"].min() - 0.01
    recommendation_jobuser.replace(-np.inf, min_value, inplace=True)
    recommendation_jobuser["score"] = recommendation_jobuser["score"] + (-1) * recommendation_jobuser["score"].min() + 0.01

    return recommendation_userjob, recommendation_jobuser


def read_data_core_20(data_items, index):
    # the data is split into chunks for parallel processing; here we read one chunk based on the index argument

    recommendation_userjob = pd.read_csv(f"./DATA/XING_DATA_INDEX/UserJob_binary_core20_10Krandomusers_1_K64603_INDEX_{index}/out.txt",
                                        header=None, names=['U_ID', 'J_ID', 'score'], dtype=str)

    recommendation_jobuser = pd.read_csv(f"./DATA/XING_DATA_INDEX/JobUser_binary_core20_10Krandomusers_1_K10000_INDEX_{index}/out.txt",
                                        header=None, names=['J_ID', 'U_ID', 'score'], dtype=str)

    recommendation_userjob["U_ID"] = pd.to_numeric(recommendation_userjob["U_ID"], errors="coerce")
    recommendation_userjob["J_ID"] = pd.to_numeric(recommendation_userjob["J_ID"], errors="coerce")
    recommendation_userjob["score"] = pd.to_numeric(recommendation_userjob["score"], errors="coerce")

    recommendation_jobuser["U_ID"] = pd.to_numeric(recommendation_jobuser["U_ID"], errors="coerce")
    recommendation_jobuser["J_ID"] = pd.to_numeric(recommendation_jobuser["J_ID"], errors="coerce")
    recommendation_jobuser["score"] = pd.to_numeric(recommendation_jobuser["score"], errors="coerce")

    recommendation_jobuser = add_sensitive_columns(recommendation_jobuser, 'J_ID', data_items,
                                                        ['country', 'is_payed'])
    recommendation_userjob = add_sensitive_columns(recommendation_userjob, 'J_ID', data_items,
                                                        ['country', 'is_payed'])
    
    return recommendation_userjob, recommendation_jobuser


def get_file_path(path, core, fusion_type, k=10, w=None, fair_metrics=None, group_cols=None, index=None):

    if core == "20":
        suffix = "N64603"

    fair_metrics_str = ""
    group_cols_str = ""

    if fair_metrics:
        fair_metrics_str = "".join([f"_{m}" for m in fair_metrics])

    if group_cols:
        group_cols_str = "".join([f"_{g}" for g in group_cols])

   
    if fusion_type == "fair_fusion_optim":
        if w is not None:
            file_path = os.path.join(path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_{fair_metrics_str}_{group_cols_str}_K{k}_w{w}", "out.txt")
        else:
            file_path = os.path.join(path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_{fair_metrics_str}_{group_cols_str}_K{k}", "out.txt")
    else:
        if w is not None:
            file_path = os.path.join(path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_alpha{w}_K{k}", "out.txt")
        else:
            file_path = os.path.join(path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_K{k}", "out.txt")
    
    if index is not None:
        file_path = file_path.replace("out.txt", f"out-{index}.txt")
    
    return file_path

def read_2sided_rec(path, core, fusion_type, data_items, k=10, w=None, fair_metrics=None, group_cols=None, index=None):
    file_path= get_file_path(path, core, fusion_type, k, w, fair_metrics, group_cols, index)

    data = pd.read_csv(file_path, header=None,  dtype=str)

    if fusion_type == "fair_fusion_optim":
        data.columns = ['U_ID', 'J_ID', 'score', 'w']
        data["w"] = pd.to_numeric(data["w"], errors="coerce")
    else:
        data.columns = ['U_ID', 'J_ID', 'score']

    data["U_ID"] = pd.to_numeric(data["U_ID"], errors="coerce")
    data["J_ID"] = pd.to_numeric(data["J_ID"], errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    if data_items is not None:
        data = add_rank_column(data, 'U_ID')
        data = add_sensitive_columns(data, 'J_ID', data_items,
                                                        ['country', 'is_payed'])
        
    return data

def merge_index_out_files(path, core, fusion_type, k=10, w=None, fair_metrics=None,group_cols=None, max_index=100):
    data = []
    for index in range(0, max_index):
        recommendation = read_2sided_rec(
            path,
            core,
            fusion_type,
            None,
            k=k if not fair_metrics else k,
            w=w,
            fair_metrics=fair_metrics,
            group_cols=group_cols,
            index=index
        )
        if fusion_type == "fair_fusion_optim":
            data.append(recommendation[['U_ID', 'J_ID', 'score', 'w']])
        else:
            data.append(recommendation[['U_ID', 'J_ID', 'score']])

    result = pd.concat(data, ignore_index=True)
    file_path = get_file_path(path, core, fusion_type, k, w, fair_metrics, group_cols, index=None)
    result.to_csv(file_path, mode='a', sep=',', index=False, header=False)