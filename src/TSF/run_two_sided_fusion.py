import pandas as pd
import os
import argparse
import numpy as np
from src.utils import add_sensitive_columns, add_rank_column
from fusion_methods import method_dict
from fair_fusion import fair_fusion_optimisation
from src.utils import process_data, add_intersectional_group_column
from src.evaluation.run_compute_metrics import run_fairness_metrics, run_compatibility_score
from src.configs import get_configs, read_args
from src.TSF.utils import read_data, read_2sided_rec, merge_index_out_files, get_file_path

def run_TSF_Fair(fusion_type, core, save_path, k, fair_metrics, group_cols, w_uf=None, data_index=None):
    # for all fairness metric and for all group cols; it is not a mapping of first group col for first metric
    
    if core == "20":
        suffix = "N64603"

    fair_metrics_str = ""
    for fair_metric in fair_metrics:
        fair_metrics_str += "_" + fair_metric
    
    group_cols_str = ""
    for group_col in group_cols:
        group_cols_str += "_" + group_col

    
    dir_path = os.path.join(save_path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_{fair_metrics_str}_{group_cols_str}_K{k}")
    if w_uf is not None:
        dir_path += f"_w{w_uf}"
    os.makedirs(dir_path, exist_ok=True)

    recommendation_userjob, recommendation_jobuser = read_data(core, data_index)
    

    result = fair_fusion_optimisation(recommendation_userjob, recommendation_jobuser, core, k, fair_metrics, group_cols, w_uf)

    result_top = result.groupby('U_ID', group_keys=False).apply(lambda x: x.nlargest(k, 'score')).reset_index(drop=True)
    result_top.to_csv(os.path.join(dir_path, f"out-{data_index}.txt"), mode='a', sep=',', index=False, header=False)

def run_TSF(fusion_type, core, save_path, k=10, w=0.5, data_index=None):
    if core == "20":
        suffix = "N64603"
    
    if fusion_type in ["rrf", "log_isr", "isr", "borda_fuse", "weighted_sum"]:
        dir_path = os.path.join(save_path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_alpha{w}_K{k}")
    else:
        dir_path = os.path.join(save_path, f"2sided_userjob_binary_core{core}_{suffix}_{fusion_type}_K{k}")
    os.makedirs(dir_path, exist_ok=True)

  
    recommendation_userjob, recommendation_jobuser = read_data(core, data_index)

    if fusion_type in ["rrf", "log_isr", "isr", "borda_fuse", "weighted_sum"]:
        result = method_dict[fusion_type](recommendation_userjob, recommendation_jobuser, w1=w)
    else:
        result = method_dict[fusion_type](recommendation_userjob, recommendation_jobuser)

    result_top = result.groupby('U_ID', group_keys=False).apply(lambda x: x.nlargest(k, 'score')).reset_index(drop=True)
    result_top.to_csv(os.path.join(dir_path, f"out-{data_index}.txt"), sep=',', index=False, header=False)


def run_evaluation(path, core, fusion_type, k=10, w=None,
    fair_metrics=None,
    group_cols=None,
    fairness=True,
    compatibility=True,
):
    
    if core == "20":
        suffix = "N64603"


    path_items = "./DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items["country"] = data_items["country"].apply(
        lambda x: "non-de" if x != "de" else x
    )

    path_users = "/home/crus/XINGInteractions/DATA/users.csv"
    data_users = pd.read_csv(path_users)
    data_users["country"] = data_users["country"].apply(
        lambda x: "non-de" if x != "de" else x
    )

   
    recommendation = read_2sided_rec(
        path,
        core,
        fusion_type,
        data_items,
        k=k if not fair_metrics else k,
        w=w,
        fair_metrics=fair_metrics,
        group_cols=group_cols,
        index=None
    )

    
    recommendation = (
            recommendation.groupby("U_ID", group_keys=False)
            .apply(lambda x: x.nlargest(k, "score"))
            .reset_index(drop=True)
        )

   
    recommendation_dir = f"./DATA/recommendation/core-{core}"

    data_test_user_job = pd.read_csv(
        os.path.join(
            recommendation_dir, "top_" + str(k), "1", "1_data_test_user_job.csv"
        )
    )
    data_train_user_job = pd.read_csv(
        os.path.join(
            recommendation_dir, "top_" + str(k), "1", "1_data_train_user_job.csv"
        )
    )


    base_name = get_file_path(
        path,
        core,
        fusion_type,        k,
        w,
        fair_metrics,
        group_cols,
        index=None,
    ).split("/")[-2]

    print(base_name)
    if fairness:
        fairness_dir = os.path.join(path, base_name, "fairness_evaluation")
        os.makedirs(fairness_dir, exist_ok=True)

        print("Computing fairness metrics...")
        run_fairness_metrics(
            recommendation,
            data_test_user_job,
            data_train_user_job,
            data_items,
            fairness_dir,
            k,
        )


    if compatibility:
        dir_path = os.path.join(path, base_name)
        fields = ['industry_id', 'discipline_id', 'career_level']
        run_compatibility_score(recommendation, data_users, data_items, dir_path, fields)


    

settings_DGI_DUP_country = [
    {
         "fair_metrics":["DGI", "DUP"],  
         "group_cols": ["country"],
    }
]
settings_DGI_country_is_payed = [
    {
         "fair_metrics":["DGI"],  
         "group_cols": ["country", "is_payed"],
    },
]

settings_DGI_country = [
    {
         "fair_metrics":["DGI"],  
         "group_cols": ["country"],
    }
]

settings_DGI_is_payed = [
    {
         "fair_metrics":["DGI"],  
         "group_cols": ["is_payed"]
    }
]

settings_DUP_country = [
    {
         "fair_metrics":["DUP"],  
         "group_cols": ["country"]
    }
]

settings_choices = {
    "settings_DGI_DUP_country": settings_DGI_DUP_country,
    "settings_DGI_country": settings_DGI_country,
    "settings_DUP_country": settings_DUP_country,
    "settings_DGI_is_payed": settings_DGI_is_payed,
    "settings_DGI_country_is_payed": settings_DGI_country_is_payed,
}


parser = argparse.ArgumentParser()
parser.add_argument("--fusion_type", type=str)
parser.add_argument("--core", type=str)
parser.add_argument("--settings", type=str)
parser.add_argument("--weight", type=str)
parser.add_argument("--index", type=str)

args = parser.parse_args()

if args.weight is not None:
    w_uf = float(args.weight)
else:
    w_uf = None

if args.settings is not None:
    settings_ = settings_choices[args.settings]

save_path = f"./DATA/core-{args.core}/2SidedJobRec"



ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
COMB_METHODS = ["comb_min", "comb_max", "comb_med"]
MAX_INDEX = 100

is_fair_optim = args.fusion_type == "fair_fusion_optim"
is_comb_method = args.fusion_type in COMB_METHODS
is_alpha_based = not is_fair_optim and not is_comb_method


# =========================
# Run TSF (per index)
# =========================
for data_index in range(MAX_INDEX):

    if is_alpha_based:
        # Global rank fusion approaches
        for alpha in ALPHAS:
            run_TSF(
                args.fusion_type,
                args.core,
                save_path,
                k=10,
                w=alpha,
                data_index=data_index,
            )

    elif is_fair_optim:
        for setting_ in settings_:
            run_TSF_Fair(
                args.fusion_type,
                args.core,
                save_path,
                10,
                setting_["fair_metrics"],
                setting_["group_cols"],
                w_uf=w_uf,
                data_index=data_index,
            )

    else:
        # comb_min / comb_max / comb_med
        run_TSF(
            args.fusion_type,
            args.core,
            save_path,
            k=10,
            w=None,
            data_index=data_index,
        )


# =========================
# Merge index output files
# =========================

if is_alpha_based:
    for alpha in ALPHAS:
        merge_index_out_files(
            save_path,
            args.core,
            args.fusion_type,
            k=10,
            w=alpha,
            fair_metrics=None,
            group_cols=None,
            max_index = MAX_INDEX
        )
elif is_fair_optim:
        for setting_ in settings_:
            merge_index_out_files(
                save_path,
                args.core,
                args.fusion_type,
                k=10,
                w=w_uf,
                fair_metrics = setting_["fair_metrics"],
                group_cols = setting_["group_cols"],
                max_index = MAX_INDEX
            )
else:
     merge_index_out_files(
        save_path,
        args.core,
        args.fusion_type,
        k=10,
        w=None,
        fair_metrics=None,
        group_cols=None,
        max_index = MAX_INDEX
    )


# =========================
# Run evaluation
# =========================
if is_alpha_based:
    for alpha in ALPHAS:
        run_evaluation(
            save_path,
            args.core,
            args.fusion_type,
            k=10,
            w=alpha,
            fair_metrics=None,
            group_cols=None,
            fairness=True,
            compatibility=True,
        )

elif is_fair_optim:
    for setting_ in settings_:
        run_evaluation(
            save_path,
            args.core,
            args.fusion_type,
            k=10,
            w=w_uf,
            fair_metrics=setting_["fair_metrics"],
            group_cols=setting_["group_cols"],
            fairness=True,
            compatibility=True,
        )

else:
    # comb_min / comb_max / comb_med
    run_evaluation(
        save_path,
        args.core,
        args.fusion_type,
        k=10,
        w=None,
        fair_metrics=None,
        group_cols=None,
        fairness=True,
        compatibility=True,
    )
