
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from src.utils import add_sensitive_columns
import numpy as np
from functools import reduce
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

const_fusion_types = [
"fair_fusion_optim__DGI__country_is_payed", 

"fair_fusion_optim__DGI__country",
"fair_fusion_optim__DUP__country",

"fair_fusion_optim__DGI__is_payed",
"fair_fusion_optim__DUP__is_payed",

"fair_fusion_optim__DGI_DUP__country", 
]

dict_metrics_group_fairness = {
    "DGI": {
        "proportional_data": "proportional_item_population",
        "proportional_population": "population_groups_items",
        "measure": "diversity"
    }, 
    "EGI": {
        "proportional_data": "proportional_item_population",
        "proportional_population": "population_groups_items",
        "measure": "exposer",
    },
    "DGI(global)": {
        "proportional_data": "proportional_item_population",
        "proportional_population": "population_groups_items__global",
        "measure": "diversity"
    }, 
    "EGI(global)": {
        "proportional_data": "proportional_item_population",
        "proportional_population": "population_groups_items__global",
        "measure": "exposer",
    },
    "DGU": {
        "proportional_data": "proportional_test",
        "proportional_population": "population_groups_query",
        "measure": "diversity",
    },
    "EGU": {
        "proportional_data": "proportional_test",
        "proportional_population": "population_groups_query",
        "measure": "exposer",
    },
    "DUP": {
        "proportional_data": "proportional_user_preference",
        "proportional_population": "clicks_item_groups_per_query(preference)",
        "measure": "diversity",
    },
    "EUP": {
        "proportional_data": "proportional_user_preference",
        "proportional_population": "clicks_item_groups_per_query(preference)",
        "measure": "exposer",
    }
}

dict_metrics_diversity = {
     "entropy": "average_entropy",
     "gini_index": "average_gini_index",
     "unique_jobs": "average_unique_jobs"
}


custom_palette = {
    "BPR": "red",  

    "TFROM(country)": "blue",
    "TFROM(premium)": "orange",
    "TFROM(country,premium)": "green",

    "ITFR(country)": "blue",
    "ITFR(premium)": "orange",
    "ITFR(country,premium)": "green",

    "FA*IR(country)": "blue",
    "FA*IR(premium)": "orange",
    "FA*IR(country,premium)": "green",

    "CFP(country)": "blue",
    "CFP(premium)": "orange",

    "PCT": "gray",
    "userfairness(country)": "blue",
    "userfairness(premium)": "orange",

    # our approach
    "TSF": "indigo",  
    "TSF - SUM": "indigo",  
    "TSF - ATT": "deeppink",
    "TSF - Min": "brown",
    "TSF - MAX": "gray", 
    "TSF - MED": "indigo", 
        
    "TSF - Borda": "orange", 
    "TSF - RRF": "green", 
    "TSF - ISR": "blue", 
    "TSF - Log ISR": "pink", 


    "TSF - Fair(RGI(country))": "yellow",
    "TSF - Fair(RGI(country,premium))": "green",
    "TSF - Fair(RGI-RUP(country))": "green",
    "TSF - Fair(RGI(premium))": "orange",

    "TSF - Fair(RUP(country))": "blue",
    "TSF - Fair(RUP(premium))": "yellow",

    }

custom_palette_bar_plot = {
    "BPR": "#e41a1c",   

    "TFROM(country)": "#1f77b4",        
    "TFROM(premium)": "#6baed6",        
    "TFROM(country,premium)": "#08306b",

    "ITFR(country)": "#2ca02c",         
    "ITFR(premium)": "#98df8a",         
    "ITFR(country,premium)": "#006400", 

    "FA*IR(country)": "#2ca02c",      
    "FA*IR(premium)": "#98df8a",        
    "FA*IR(country,premium)": "#145214", 

    "CP-Fair(country)": "#ffbf00",   
    "CP-Fair(premium)": "#ffe680",   

    "userfairness(country)": "#8c564b", 
    "userfairness(premium)": "#c49c94", 

    "PCT": "#7f7f7f",

    "TSF": "#800080",          
    "TSF - ATT": "#ff69b4",         
}



def get_eval_dirs(method, root_dir, fusion_type, core, k):
    dirs_all = os.listdir(root_dir)

    # Match both "_k_10" and "K10" anywhere
    k_patterns = [f"_k_{k}", f"K{k}"]

    if fusion_type is None:
        dirs_ = [
            dir_ for dir_ in dirs_all
            if f"core{core}" in dir_
            and method in dir_
            and any(kp in dir_ for kp in k_patterns)
        ]

        if method == "2sided":
            dirs_ = [dir_ for dir_ in dirs_ if "ATT" not in dir_]
    else:
        # fusion_type can appear anywhere, K can appear anywhere
        dirs_ = [
            dir_ for dir_ in dirs_all
            if f"core{core}" in dir_
            and method in dir_
            and fusion_type in dir_
            and any(kp in dir_ for kp in k_patterns)
        ]


        # ISR special case: remove log_ dirs
        if fusion_type == "isr":
            dirs_ = [dir_ for dir_ in dirs_ if "log_" not in dir_]
    
    return dirs_

def get_recsys_dirs(method, root_dir, fusion_type, core, k):
    if fusion_type is None:
        if method == "2sided":
            dirs_path = os.path.join(root_dir, "2SidedJobRec")
            dirs_ = [os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path) if "core" + str(core) in dir_ and method in dir_ and f"_K{k}" in dir_]
        elif method == "UserJob":
            dirs_path = os.path.join(root_dir, "Baseline_BPR")
            dirs_ = [os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path) if "core" + str(core) in dir_ and "UserJob" in dir_ ]
        elif method == "2sided_ATT":
            dirs_path = os.path.join(root_dir, f"core-{core}", "2SidedJobRec")
            dirs_ = [os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path) if "core" + str(core) in dir_ and "ATT" in dir_ ]
        else:
            dirs_path = os.path.join(root_dir, f"Baseline_{method}")
            dirs_ = [os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path) if "core" + str(core) in dir_ and method in dir_ ]
    else:
        dirs_path = os.path.join(root_dir)
        dirs_ = [
        os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path)
        if "core" + str(core) in dir_
        and "2sided" in dir_
        and f"{fusion_type}_K" in dir_
        ]

        if len(dirs_) == 0:
            dirs_ = [
        os.path.join(dirs_path, dir_) for dir_ in os.listdir(dirs_path)
        if "core" + str(core) in dir_
        and "2sided" in dir_
        and f"{fusion_type}" in dir_
        ]
        if fusion_type == "isr":
            dirs_ = [os.path.join(dirs_path, dir_) for dir_ in dirs_ if "log_" not in dir_] 
    
    return dirs_ 

def get_params(dir_, fusion_type): 
    split_key = []
    if "alpha" in dir_:
        split_key.append("alpha")
    if "param" in dir_:
        split_key.append("param")
    if "w" in dir_:
        split_key.append("w")
    if "N" in dir_ and fusion_type is None:
        split_key.append("N")
    if len(split_key) == 0:
        split_key = None

    if split_key is not None:
        params = []
        for s_key in split_key:
            if fusion_type is not None:
                params.append(dir_.split(f"{s_key}")[1].split("_")[0])
            else:
                if s_key == "N":
                    params.append(dir_.split(f"{s_key}")[1].split("_")[0])
                else:
                    params.append(dir_.split(f"{s_key}_")[1].split("_")[0])
        if len(params) > 1:
            params = "(" + ",".join(params) + ")"
        else:
            params = params[0]
    else:
        params = None

    return params

def get_groups(dir_):
    if "groups" in dir_:   
        groups = dir_.split("groups_")[1].split("_")[0]
       
        if "is_payed" in dir_:
            groups = groups.replace("is", "ispayed")
    else:
        groups = None
    return groups

def get_eval_dir(root_dir, dir_, fusion_type):
    if fusion_type is None:
        eval_dir = os.path.join(root_dir, dir_)
    else:
        eval_dir = os.path.join(root_dir, dir_, "fairness_evaluation")
    return eval_dir

def get_recsys_file(root_dir, dir_, fusion_type):
    _recsys_file = os.path.join(root_dir, dir_, "out.txt")
    if not os.path.exists(_recsys_file):
        _recsys_file = os.path.join(root_dir, dir_, "out-1.txt")
    
    if not os.path.exists(_recsys_file):
        _recsys_file = os.path.join(root_dir, dir_, "out-1.txt")

    return _recsys_file

def get_method_name(method, groups, fusion_type):
    if groups is not None:
        method_name = f"{method}({groups})"
    elif fusion_type is not None:
        method_name = f"{method}_{fusion_type}"
    else:
        method_name = method
    return method_name

def read_metric(eval_dir, metric, group, per_group):
    if metric in dict_metrics_group_fairness:
            if group == "__country__is_payed":
                file_name = dict_metrics_group_fairness[metric]["proportional_population"] + "_per_group_items.csv"
                metric_file_path = os.path.join(eval_dir, dict_metrics_group_fairness[metric]["proportional_data"], group, dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" +  file_name)
                metric_value = pd.read_csv(metric_file_path)['eval'].mean()
            elif group == "intersectional__country__is_payed":
                file_name = dict_metrics_group_fairness[metric]["proportional_population"] + ".csv_per_group_items.csv"
                metric_file_path = os.path.join(eval_dir, dict_metrics_group_fairness[metric]["proportional_data"], group.replace("intersectional", ""), "intersectional_" +  dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" +  file_name)
                data_eval = pd.read_csv(metric_file_path)
                metric_value = data_eval.groupby("__country__is_payed").apply(lambda x: x["eval"].mean()).reset_index()
                metric_value =  metric_value.rename(columns={metric_value.columns[-1]:metric})
            elif per_group:
                file_name = dict_metrics_group_fairness[metric]["proportional_population"] + "_per_group_items.csv"
                metric_file_path = os.path.join(eval_dir, dict_metrics_group_fairness[metric]["proportional_data"], group, dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" +  file_name)
                data_eval = pd.read_csv(metric_file_path)
                if not per_UID:
                    metric_value= data_eval.groupby(group).apply(lambda x: x["eval"].mean()).reset_index()
                    metric_value =  metric_value.rename(columns={metric_value.columns[-1]:metric})
                else:
                    metric_value = data_eval
            else:
                file_name = dict_metrics_group_fairness[metric]["proportional_population"] + ".csv"
                metric_file_path = os.path.join(eval_dir, dict_metrics_group_fairness[metric]["proportional_data"], group , dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" +  file_name)
                metric_value = pd.read_csv(metric_file_path)['0'].values[0]

    elif metric in dict_metrics_diversity:
            metric_file_path = os.path.join(eval_dir, dict_metrics_diversity[metric], metric + "_all.csv")
            metric_value = pd.read_csv(metric_file_path)[metric].values[0]

    return metric_value

def read_compatibility(eval_dir, metric):
    compatibility_analysis = False
    if metric == "compatibility_score":
        compatibility_file = os.path.join(root_dir, eval_dir, "compatibility_wc", "compatibility_user_job.csv")
    elif metric in ["compatibility_added_de", "compatibility_added_non-de", "compatibility_removed_de", "compatibility_removed_non-de"]:
        if "removed" in metric:
            metric_group = metric.split("_")[-1]
            metric_file = metric.replace(metric_group, f"_{metric_group}")
        else:
            metric_file = metric
        compatibility_file = os.path.join(root_dir, eval_dir, "country_rec_analysis_wc", f"{metric_file}.csv")
    elif metric in ["compatibility_added_0", "compatibility_added_1", "compatibility_removed_0", "compatibility_removed_1"]:
        if "removed" in metric:
            metric_group = metric.split("_")[-1]
            metric_file = metric.replace(metric_group, f"_{metric_group}")
        else:
            metric_file = metric
        compatibility_file = os.path.join(root_dir, eval_dir, "is_payed_rec_analysis_wc", f"{metric_file}.csv")
    elif "__" in metric:
        compatibility_file = os.path.join(root_dir, eval_dir, "compatibility", "compatibility_user_job.csv")
        compatibility_analysis = True
    if not os.path.exists(compatibility_file):
        return -1
    data = pd.read_csv(compatibility_file)
    if not compatibility_analysis:
        return data["match_score"]
    else:
        group = metric.split("__")[1]
        group_value = metric.split("__")[2]
        raw_str = data.loc[0, group]

        tokens = [t for t in raw_str.split() if t]

        pairs = [(tokens[3], float(tokens[4])), (tokens[6], float(tokens[7]))]

        item_df = pd.DataFrame(pairs, columns=["group_name", "value"])
        return item_df[item_df["group_name"] == group_value]["value"]


def read_recommendation_file(method, root_dir, core, fusion_type=None, k=10):

    path_items = "/home/crus/XINGInteractions/DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items['country'] = data_items['country'].apply(lambda x: 'non-de' if x != 'de' else x)

    dirs__recsys = get_recsys_dirs(method, root_dir, fusion_type, core, k)
    
    metric_values = []
    dirs__eval = get_eval_dirs(method, root_dir, fusion_type, core, k)
    if len(dirs__eval) == 0:
        dirs__eval = get_eval_dirs(method, os.path.join(root_dir, "fairness_evaluation"), fusion_type, core, k)
   
    metric_values = []
    dirs__recsys = sorted(dirs__recsys)
    dirs__eval = sorted(dirs__eval)
   
    for dir_recsys, dir_eval in zip(dirs__recsys, dirs__eval):
        params = get_params(dir_eval, fusion_type)
        groups = get_groups(dir_recsys)
        recs_file = get_recsys_file(root_dir, dir_recsys, fusion_type)
        if not os.path.exists(recs_file):
            continue
       
        data = pd.read_csv(recs_file, sep=",", header=None, names=["U_ID", "J_ID", "S"])
        data = data.groupby("U_ID").apply(lambda x: x.sort_values("S", ascending=False)[:k]).reset_index(drop=True)

        data = add_sensitive_columns(data, 'J_ID', data_items,
                                                    ['country', 'is_payed'])
        
        avg_country_per_user = data.groupby(['U_ID', 'country']).size().reset_index().rename(columns={0: 'count'}).groupby("country")['count'].mean().reset_index()#/k
        avg_premium_per_user = data.groupby(['U_ID', 'is_payed']).size().reset_index().rename(columns={0: 'count'}).groupby("is_payed")['count'].mean().reset_index()#/k
        avg_country = data.groupby(['country']).size().reset_index().rename(columns={0: 'count'})#.groupby("country")['count']/(k*data['U_ID'].nunique())
        avg_premium = data.groupby(['is_payed']).size().reset_index().rename(columns={0: 'count'})#.groupby("is_payed")['count']/(k*data['U_ID'].nunique())

        
        method_name = get_method_name(method, groups, fusion_type)
        metric_values.append(pd.DataFrame.from_dict({
        "method": [method_name],
        "params": [params],
        "avg_country_per_user_de": [avg_country_per_user[avg_country_per_user["country"] == "de"]["count"].values[0] / k],
        "avg_country_per_user_nonde": [avg_country_per_user[avg_country_per_user["country"] == "non-de"]["count"].values[0] / k],
        "avg_premium_per_user_0": [avg_premium_per_user[avg_premium_per_user["is_payed"].astype(str) == "0"]["count"].values[0] / k],
        "avg_premium_per_user_1": [avg_premium_per_user[avg_premium_per_user["is_payed"].astype(str) == "1"]["count"].values[0] / k],
        "avg_country_de": [avg_country[avg_country["country"] == "de"]["count"].values[0] / (k*data['U_ID'].nunique())],
        "avg_country_nonde": [avg_country[avg_country["country"] == "non-de"]["count"].values[0] / (k*data['U_ID'].nunique())],
        "avg_premium_0": [avg_premium[avg_premium["is_payed"].astype(str) == "0"]["count"].values[0] / (k*data['U_ID'].nunique())],
        "avg_premium_1": [avg_premium[avg_premium["is_payed"].astype(str) == "1"]["count"].values[0] / (k*data['U_ID'].nunique())]
    }))


    df_metric_values = pd.concat(metric_values)
    return df_metric_values


def read_evaluation_file(method, root_dir, core, group, metric, per_group=False, per_UID=False, fusion_type=None, k=10):
    # dirs__recsys = get_recsys_dirs(method, "/".join(root_dir.split("/")[:-1]), fusion_type, core, k)
    dirs__eval = get_eval_dirs(method, root_dir, fusion_type, core, k)
    metric_values = []
    for dir_eval in dirs__eval:
        params = get_params(dir_eval, fusion_type)
        groups = get_groups(dir_eval)

        if metric in ['precision','recall','ndcg','coverage','gini']: 
            for user_group in ["All", "German", "NonGerman", "Premium", "NonPremium"]:
                metric_value = read_utility_diversity_metrics(method, fusion_type, core, metric, k, user_group=user_group, item_group=groups, params=params)

                method_name = get_method_name(method, groups, fusion_type)
                df_metric_values = pd.DataFrame.from_dict({
                "method": [method_name],
                "params": [params],
                "groups": [groups],
                f"{metric}({user_group})": [float(metric_value)]
                })

                metric_values.append(df_metric_values)
        elif "compatibility" in metric:
            eval_dir = get_eval_dir(root_dir, dir_eval, fusion_type)
            if fusion_type is not None:
                eval_dir = eval_dir.replace("fairness_evaluation", "")
            metric_value = read_compatibility(eval_dir, metric)
        else:
            eval_dir = get_eval_dir(root_dir, dir_eval, fusion_type)
            metric_value = read_metric(eval_dir, metric, group, per_group)


        if metric not in ['precision','recall','ndcg','coverage','gini']: 
            method_name = get_method_name(method, groups, fusion_type)
            

            if not "intersectional" in group and not per_group:                
                df_metric_values = pd.DataFrame.from_dict({
                    "method": [method_name],
                    "params": [params],
                    "groups": [groups],
                    metric: [float(metric_value)]
                })
            else:
                df_metric_values = metric_value
                df_metric_values["method"] = [method] * len(df_metric_values)
                df_metric_values["params"] = [params] * len(df_metric_values)
                df_metric_values["groups"] = [groups] * len(df_metric_values)

            metric_values.append(df_metric_values)
    
    if len(metric_values) == 0:
        return None
    df_metric_values = pd.concat(metric_values)
    return df_metric_values


def read_utility_diversity_metrics(method, fusion_type, core, metric, k, user_group, item_group, params):
    # item_group is the method_group - the group param used to run a method
    # user_group is the user-fairness evaluation, meaning precision for user group X
    columns = {
        "bpr": ['attr','precision','recall','ndcg','coverage','gini'],
        "TFROM": ['attr','criteria','feature','N','precision','recall','ndcg','coverage','gini'],
        "ITFR": ['attr','feature','precision','recall','ndcg','coverage','gini'],
        "PCT": ['attr','alpha','precision','recall','ndcg','coverage','gini'],
        "CFP": ['attr','feature','N','epsilon','precision','recall','ndcg','coverage','gini'],
        "2sided_ATT": ['attr','precision','recall','ndcg','coverage','gini'],
        "2sided": ['attr','alpha','precision','recall','ndcg','coverage','gini'],
        "weighted_sum": ['attr','alpha','precision','recall','ndcg','coverage','gini'],
        "userfairness": ['attr', 'feature', 'N','precision','recall','ndcg','coverage','gini'],
        "FairStar": ['attr','feature','N','p', 'a', 'precision','recall','ndcg','coverage','gini']
    }

    dir_path = os.path.join(root_dir, "performance_evaluation")
    files = os.listdir(dir_path)
    
    
    if method == "UserJob":
        method = "bpr"
    
    
    file_match = [file for file in files if f"core{core}" in file and f"K{k}" in file] 
    
    if fusion_type is not None and fusion_type != "weighted_sum":
        if 'fair_fusion_optim' not in fusion_type:
            file_match = [file for file in file_match if "rankfusion" in file]
        else:
            file_match = [file for file in file_match if "fairfusion" in file]

    elif "ATT" in method:
        file_match = [file for file in file_match if "ATT" in file]
    else:
        file_match = [file for file in file_match if method.lower() in file and "ATT" not in file]

    if len(file_match) > 1:
        print(file_match)
        raise Exception("My ERROR: too many files")
    
    if len(file_match) == 1:
        file = file_match[0]
        results = pd.read_csv(os.path.join(dir_path, file), sep="\t", header=None)
        if fusion_type and fusion_type != "weighted_sum":
            if "fair_fusion_optim" in fusion_type:
                columns = ['attr', "fusion_type", 'precision','recall','ndcg','coverage','gini']
            else:
                columns = ['attr', "fusion_type", 'alpha','precision','recall','ndcg','coverage','gini']
        else:
            columns = columns[method]
        results.columns = columns

       
        results = results[results["attr"] == user_group]
        
        if "feature" in columns:
            if item_group == "country":
                item_group = "c"
            elif item_group == "premium" or item_group == "ispayed":
                item_group = "p"
            elif item_group == "countryispayed" or item_group == "countrypremium":
                item_group = "cp"
            results = results[results['feature'] == item_group]
        if "alpha" in columns:
            if "fusion_type" in columns:
                if "min" in fusion_type or "max" in fusion_type:
                    params = 0
            results = results[results['alpha'].astype(float) == float(params)]
        if "epsilon" in columns:
            results = results[results['epsilon'].astype(float) == float(params)]

        if method == "FairStar":
            p_params = params.split(",")[0].replace("((", "")
            a_params = params.split(",")[1].replace(")", "")
            N_params = params.split(",")[2].replace(")", "")

            results = results[results['p'].astype(float) == float(p_params)]
            results = results[results['a'].astype(float) == float(a_params)]
       
        if "N" in columns:
            if method == "CFP":
                N = "200"
            elif method == "userfairness":
                N = params.split(",")[1].split(")")[0]
            elif method == "FairStar":
                N = N_params
            else:
                N = params
            results = results[results['N'].astype(int) == int(N)]
        if "fusion_type" in columns:
            if "fair_fusion_optim" in fusion_type:
                results["fusion_type"] = "fair_fusion_optim__" + results["fusion_type"].astype(str)# + "__" + results["fusion_group"].astype(str)
            if "min" in fusion_type or "max" in fusion_type:
                results["fusion_type"]  = "comb_" + results["fusion_type"].astype(str)
            results = results[results["fusion_type"] == fusion_type]
            
        if len(results) > 1:
            print(results)
            raise Exception("MY ERROR: too many results")
        if len(results) == 0:
            return -1
        return results[metric].values[0]
    else:
        return -1


def read_eval_results_methods(root_dir, core, fairness_metrics, k):
    root_dir_fusion = os.path.join(root_dir, f"core-{core}", "2SidedJobRec")
    root_dir = os.path.join(root_dir, "fairness_evaluation")

    methods = ["UserJob","2sided"]#["2sided_ATT", "UserJob", "2sided", "TFROM", "ITFR", "CFP", "userfairness", "FairStar"]
    utility_metrics = ['precision', 'recall', 'ndcg', 'coverage', 'gini']

    if k == 10:
        fusion_types = ["weighted_sum"] #, "isr", "rrf", "borda_fuse", "comb_max", "comb_min"]
        fusion_types.extend(const_fusion_types)
    else:
        fusion_types = []

    all_results = []

    for fairness_metric in fairness_metrics:
        for group in ["country", "is_payed"]:
            group_name = "premium" if group == "is_payed" else group

            # skip is_payed for utility metrics (we don’t need recall(premium), etc.)
            if fairness_metric in utility_metrics and group != "country":
                continue

            results_metric = []

            # methods
            for method in methods:
                try:
                    df = read_evaluation_file(method, root_dir, core, group, fairness_metric, k=k)
                except FileNotFoundError:
                    continue
                if df is None:
                    continue
                if df.empty:
                    continue

                # column naming
                if fairness_metric in utility_metrics:
                    for user_group in ["All", "German", "NonGerman", "Premium", "NonPremium"]:
                        col_name = f"{fairness_metric}({user_group})"
                        temp_df = df[["method", "params", "groups", col_name]].copy()
                        temp_df = temp_df.rename(columns={col_name: "value"})
                        temp_df["metric"] = col_name
                        results_metric.append(temp_df[["method", "params", "groups", "metric", "value"]])
                elif "compatibility" in fairness_metric or "p_" in fairness_metric:
                    col_name = fairness_metric  # no group suffix
                    df = df.rename(columns={fairness_metric: "value"})
                    df["metric"] = col_name
                    results_metric.append(df[["method", "params", "groups", "metric", "value"]])
                else:
                    col_name = f"{fairness_metric}({group_name})"
                    df = df.rename(columns={fairness_metric: "value"})
                    df["metric"] = col_name
                    results_metric.append(df[["method", "params", "groups", "metric", "value"]])

            # fusion types
            for fusion_type in fusion_types:
                try:
                    df = read_evaluation_file("2sided", root_dir_fusion, core, group, fairness_metric, fusion_type=fusion_type, k=k)
                except FileNotFoundError:
                    continue
                if df is None:
                    print("No metrics for thie method!")
                    continue

                if df.empty:
                    continue

                if fairness_metric in utility_metrics:
                    for user_group in ["All", "German", "NonGerman", "Premium", "NonPremium"]:
                        col_name = f"{fairness_metric}({user_group})"
                        temp_df = df[["method", "params", "groups", col_name]].copy()
                        temp_df = temp_df.rename(columns={col_name: "value"})
                        temp_df["metric"] = col_name
                        results_metric.append(temp_df[["method", "params", "groups", "metric", "value"]])
                elif "compatibility" in fairness_metric or "p_" in fairness_metric:
                    col_name = fairness_metric  # no group suffix
                    df = df.rename(columns={fairness_metric: "value"})
                    df["metric"] = col_name
                    results_metric.append(df[["method", "params", "groups", "metric", "value"]])
                else:
                    col_name = f"{fairness_metric}({group_name})"
                    df = df.rename(columns={fairness_metric: "value"})
                    df["metric"] = col_name
                    results_metric.append(df[["method", "params", "groups", "metric", "value"]])

            if results_metric:
                all_results.append(pd.concat(results_metric, ignore_index=True))

    if not all_results:
        return pd.DataFrame()

    long_df = pd.concat(all_results, ignore_index=True)
    long_df["params"] = long_df["params"].fillna("None")
    long_df["groups"] = long_df["groups"].fillna("None")
    all_index = (
        long_df[["method", "params", "groups"]]
        .drop_duplicates()
        .set_index(["method", "params", "groups"])
    )

    wide_df = long_df.pivot_table(
        index=["method", "params", "groups"],
        columns="metric",
        values="value",
        aggfunc="first"  
    )

    wide_df = wide_df.reindex(all_index.index)

    wide_df = wide_df.reset_index()

    wide_df.columns.name = None  

    print("Methods in wide_df:", wide_df["method"].unique())

    return wide_df



def generate_results_file(root_dir, core, k):
    metrics = ["DGI", "DGI(global)", "DUP"]# "ndcg", "precision", "recall", "coverage", "gini"]    
    compatibility_metrics = ["compatibility_score", "compatibility_added_de", "compatibility_added_non-de", "compatibility_removed_de", "compatibility_removed_non-de", "compatibility_added_0", "compatibility_added_1", "compatibility_removed_0", "compatibility_removed_1"]

    #metrics.extend(compatibility_metrics)
    results = read_eval_results_methods(root_dir, core, metrics, k)
    results.to_csv(os.path.join(root_dir, f"core-{core}", f"evaluation_results_core{core}_K{k}.csv"))



def init_plot(plots_per_row, fairness_metrics, num_rows=None):
    if num_rows == None:
        num_rows = math.ceil(len(fairness_metrics) / plots_per_row)
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(3 * plots_per_row, 4 * num_rows), sharex=False, sharey=False)
    axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [axes]
    
    return fig, axes #[:len(fairness_metrics)]


def create_legend(fig, axes):
    axes_list = np.ravel(np.atleast_1d(axes)).tolist()

    empty_axes = []
    non_empty_axes = []

    for ax in axes_list:
        if (
            len(ax.lines) == 0
            and len(ax.patches) == 0
            and len(ax.collections) == 0
            and len(ax.images) == 0
            and len(ax.containers) == 0
        ):
            empty_axes.append(ax)
        else:
            non_empty_axes.append(ax)

    legend_ax_pos = None
    if empty_axes:
        legend_ax_pos = empty_axes[-1].get_position()
        for ax in empty_axes:
            fig.delaxes(ax)

    handles, labels = [], []
    for ax in non_empty_axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        if ax.get_legend():
            ax.get_legend().remove()

    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    labels_sorted = sorted(unique.keys())
    handles_sorted = [unique[l] for l in labels_sorted]

    # Place legend exactly in empty subplot space
    if legend_ax_pos is not None:
        fig.legend(
            handles_sorted,
            labels_sorted,
            title="Method",
            fontsize=20,
            title_fontsize=17,
            frameon=False,
            loc="center",
            bbox_to_anchor=(
                legend_ax_pos.x0 + legend_ax_pos.width / 2,
                legend_ax_pos.y0 + legend_ax_pos.height / 2,
            ),
            bbox_transform=fig.transFigure,
            ncol=1,
            handletextpad=0.4,
            columnspacing=0.8,
        )

    fig.tight_layout()



def save_fig(core, file_name):
    plt.savefig(os.path.join(f"/home/crus/XINGInteractions/DATA/core-{core}/plots", file_name), bbox_inches="tight")  


def plot_results_ax(fig, ax, results, metric_x, metric_y, params=False):
    params_num = pd.to_numeric(results["params"], errors="coerce")
    mask_line_plot = (~params_num.isna())
    results_line_plot = results[mask_line_plot]
    results_line_plot = results_line_plot.reset_index(drop=True)
  
    if len(results_line_plot) != 0:
        sns.lineplot(
                    data=results_line_plot,
                    x=metric_x,
                    y=metric_y,
                    hue="method",
                    palette=custom_palette,
                    ax=ax,
                    estimator=None,  
                    errorbar=None
        )
    

    if "DGI" in metric_x or "DUP" in metric_x:
        metric_x = metric_x.replace("D", "R")
    else:
        if "α" not in metric_x and "ndcg" not in metric_x:
            metric_x = metric_x.replace("_", " ")
            metric_x = metric_x.title()

    if "DGI" in metric_y or "DUP" in metric_y:
        metric_y = metric_y.replace("D", "R")
    else:
        if "α" not in metric_y and "ndcg" not in metric_y:
            metric_y = metric_y.replace("_", " ")
            metric_y = metric_y.title()

    if "ndcg" in metric_x:
        metric_x = metric_x.upper()

    if "ndcg" in metric_y:
        metric_y = metric_y.upper()

    ax.set_xlabel(metric_x, fontsize=23)
    ax.set_ylabel(metric_y, fontsize=23)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))  # 2 decimals
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))  # 1 decimal   

    ax.tick_params(axis="x", which="major", pad=15)  # default is ~5
    ax.tick_params(axis="y", which="major", pad=15)  # default is ~5

    plt.tight_layout()
   
    
def format_naming(results):
    method_names = results["method"].unique()
    
    dict_method_name = {
        'FairStar(country)': "FA*IR(country)",
        'FairStar(ispayed)': "FA*IR(premium)",
        'FairStar(countryispayed)': "FA*IR(country,premium)",
        'UserJob': "BPR",

        '2sided_weighted_sum': "TSF - SUM",
        '2sided': "TSF",
        '2sided_ATT': "TSF - ATT",
        '2sided_log_isr': "TSF - Log ISR",
        '2sided_isr': "TSF - ISR",
        '2sided_comb_min': "TSF - Min", 
        '2sided_borda_fuse': "TSF - Borda",
        '2sided_rrf': "TSF - RRF", 
        '2sided_comb_max': "TSF - MAX", 
        '2sided_comb_med': "TSF - MED",

        'CFP(premium)': "CP-Fair(premium)",
        "CFP(country)": "CP-Fair(country)",
        'TFROM(ispayed)': "TFROM(premium)",
        'TFROM(countryispayed)': 'TFROM(country,premium)',
        
        'ITFR(premium)': 'ITFR(premium)',
        'ITFR(countrypremium)': 'ITFR(country,premium)',


        "2sided_fair_fusion_optim__DGI__country":"TSF - Fair(RGI(country))",
        "2sided_fair_fusion_optim__DGI__is_payed":"TSF - Fair(RGI(premium))",
        "2sided_fair_fusion_optim__DGI__country_is_payed": "TSF - Fair(RGI(country,premium))",
        '2sided_fair_fusion_optim__DGI_DUP__country': "TSF - Fair(RGI-RUP(country))",

        "2sided_fair_fusion_optim__DUP__country":"TSF - Fair(RUP(country))",
        "2sided_fair_fusion_optim__DUP__is_payed":"TSF - Fair(RUP(premium))",
        
    }
    
    
    results["method"] = results["method"].apply(lambda x: dict_method_name[x] if x in dict_method_name else x)
    return results


def add_method_type(results):
    method_names = results["method"].unique()

    dict_method_name = {
        'FairStar(country)': "item",
        'FairStar(ispayed)': "item",
        'FairStar(countryispayed)': "item",
        'UserJob': "BPR",

        '2sided': "fusion_score",
        '2sided_ATT': "fusion_score",
        '2sided_log_isr': "fusion_rank",
        '2sided_isr': "fusion_rank",
        '2sided_comb_min': "fusion_rank_score", 
        '2sided_borda_fuse': "fusion_rank",
        '2sided_rrf': "fusion_rank", 
        '2sided_comb_max': "fusion_rank_score", 
        '2sided_comb_med': "fusion_rank_score",

        "CFP(country)": "two-sided",
        "CFP(premium)": "two-sided",
        "userfairness(country)": "user",
        "userfairness(premium)": "user",

        "PCT": "two-sided",

        'TFROM(country)': "two-sided",
        'TFROM(ispayed)': "two-sided",
        'TFROM(countryispayed)': "two-sided",
        
        'ITFR(country)': "two-sided",
        'ITFR(premium)': "two-sided",
        'ITFR(countrypremium)': "two-sided",
        
        "2sided_fair_fusion_optim__DGI__country":"fusion_learn",
        "2sided_fair_fusion_optim__DGI__is_payed":"fusion_learn",
        "2sided_fair_fusion_optim__DGI__country_is_payed": "fusion_learn",

        "2sided_fair_fusion_optim__DUP__country":"fusion_learn",
        '2sided_fair_fusion_optim__DGI_DUP__country': "fusion_learn"

    }

    results["method_type"] = results["method"].apply(lambda x: dict_method_name[x] if x in dict_method_name else x)
    return results


def read_results_file(root_dir, core, k):
    results = pd.read_csv(os.path.join(root_dir, f"core-{core}", f"evaluation_results_core{core}_K{k}.csv"))
    
    # for TFROM, userfairness select only N=200
    results_TFROM = results[np.logical_and(results["method"].str.contains("TFROM"), results["params"] == "200")]
    results_userfairness = results[np.logical_and(results["method"].str.contains("userfairness"), results["params"] == f"(precision@{k},200)")]
    mask = results["params"].str.contains(r",1000\)$", regex=True)
    results_fairstar = results[
        np.logical_and(results["method"].str.contains("FairStar"), mask)
    ]
    # mask_rest = np.logical_not(np.logical_or(results["method"].str.contains("TFROM"), results["method"].str.contains("userfairness")))
    mask_rest = np.logical_not(np.logical_or.reduce([
    results["method"].str.contains("TFROM"),
    results["method"].str.contains("userfairness"),
    results["method"].str.contains("FairStar"),
    ]))

    # exclude rank_fusion as they have very low performance results
    # exclude ITFR as it is an in-processing approach
    
    for method in ["2sided_isr", "2sided_log_isr", "2sided_rrf", "2sided_borda_fuse", "ITFR", "2sided_comb_max", "2sided_comb_min"]:
        mask_rest = np.logical_and(mask_rest, np.logical_not(results["method"].str.contains(method)))
    results = results[mask_rest]

    results = pd.concat([results, results_TFROM, results_userfairness, results_fairstar])
   
    return results


def get_best_params(results, core, k=10):
    params_num = pd.to_numeric(results["params"], errors="coerce")

    if core == "20" and k==10:
        mask = np.logical_and(results["method"] == "2sided", params_num == 0.5) 


        mask = np.logical_or(mask, np.logical_and(results["method"] == "FairStar(ispayed)", results["params"] == "((0.6, 0.15),1000)"))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "FairStar(country)", results["params"] == "((0.6, 0.1),1000)"))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "FairStar(countryispayed)", results["params"] == "((0.6, 0.05),1000)"))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "2sided_isr", params_num == 0.7))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "2sided_log_isr", params_num == 0.8))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "2sided_borda_fuse", params_num == 0.8))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "2sided_rrf", params_num == 0.8))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "CFP(country)", params_num == 0.5))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "CFP(premium)", params_num == 0.5))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "PCT", params_num == 0.4))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "userfairness(country)", results["params"] == f"(precision@{k},200)"))
        mask = np.logical_or(mask, np.logical_and(results["method"] == "userfairness(premium)", results["params"] == f"(precision@{k},200)"))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "TFROM(country)", params_num == 200))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "TFROM(ispayed)", params_num == 200))

        mask = np.logical_or(mask, np.logical_and(results["method"] == "TFROM(countryispayed)", params_num == 200))
        

    mask = np.logical_or(mask, results["params"].isna())
    mask = np.logical_or(mask, results["params"] == "None")
    mask = np.logical_or(mask, results["params"] == "")
    return results[mask]

def add_params_expanded(results):
    param_values = np.arange(0, 1.1, 0.1)  # [0.1, ..., 0.9]

    subset = results[results["params"].isna()]
    expanded = (
        subset.loc[subset.index.repeat(len(param_values))]
        .assign(params=param_values.tolist() * len(subset))
    )
    
    return expanded

def plot_metric_vs_alpha(root_dir, core, k=10, rank_fusion=False):
    # -----------------------
    # Load & preprocess data
    # -----------------------
    results_path = os.path.join(
        root_dir, f"core-{core}", f"evaluation_results_core{core}_K{k}.csv"
    )
    results = pd.read_csv(results_path)
    results = add_method_type(results)

    # -----------------------
    # Helper functions
    # -----------------------
    def with_params(df):
        df = add_params_expanded(df)
        df["α"] = df["params"]
        return df

    def filter_methods(df, patterns, regex=False):
        if regex:
            mask = df["method"].str.contains(patterns, regex=True)
        else:
            mask = df["method"].isin(patterns)
        return df[mask]

    # -----------------------
    # Baselines
    # -----------------------
    results_baselines = with_params(
        results[results["method"].str.contains("UserJob")]
    )

    # -----------------------
    # 2-sided weighted sum
    # -----------------------
    results_2sided = results[
        results["method"].str.contains("2sided_weighted_sum")
    ]
    results_2sided["α"] = results_2sided["params"]

    # -----------------------
    # Rank fusion variants
    # -----------------------
    results_rank_fusion = results[
        (results["method_type"] == "fusion_rank") &
        (~results["method"].str.contains("log"))
    ]

    results_rank_fusion_score = with_params(
        results[results["method_type"] == "fusion_rank_score"]
    )

    # -----------------------
    # Fair fusion optimization
    # -----------------------
    # fair_optim_methods = [
    #     "2sided_fair_fusion_optim__DGI__country",
    #     "2sided_fair_fusion_optim__DUP__country",
    # ]

    # fair_optim_dual_methods = [
    #     "2sided_fair_fusion_optim__DGI_DUP__country",
    # ]

    fair_optim_methods = [
        "2sided_fair_fusion_optim__DGI__country",
        "2sided_fair_fusion_optim__DGI__is_payed",
    ]

    fair_optim_dual_methods = [
        "2sided_fair_fusion_optim__DGI__country_is_payed",
    ]


    mask_fair_optim = (
        results["method"].str.contains("2sided") &
        results["method"].str.contains("fair_fusion_optim")
    )

    results_fair_optim = with_params(
        filter_methods(results[mask_fair_optim], fair_optim_methods)
    )

    results_fair_dual_optim = filter_methods(results[mask_fair_optim], fair_optim_dual_methods)
    results_fair_dual_optim["α"] = results_fair_dual_optim["params"]
    results_fair_optim = pd.concat([results_fair_optim, results_fair_dual_optim])
    # -----------------------
    # Assemble plot dataframe
    # -----------------------
    if rank_fusion:
        results_plot = pd.concat([
            results_2sided,
            results_baselines,
            results_rank_fusion,
            results_rank_fusion_score,
        ])
    else:
        results_plot = pd.concat([
            results_2sided,
            results_baselines,
            results_fair_optim,
        ])

    results_plot = format_naming(results_plot)
    results_plot["α"] = pd.to_numeric(results_plot["α"], errors="coerce")

    # -----------------------
    # Metrics to plot
    # -----------------------
    fair_metrics = [
        "DGI(premium)",
        "DGI(country)",
        "DGI(global)(country)",
        "DUP(country)",
    ]

    metrics = fair_metrics

    # -----------------------
    # Plot setup
    # -----------------------
    fig, axes = init_plot(
        plots_per_row=len(metrics),
        fairness_metrics=metrics,
        num_rows=2
    )

    name_metric = "base_metric_rank_fusion" if rank_fusion else "base_metric"
    file_name = f"2sided_fair_optim_alpha_vs_{name_metric}_K{k}.png"

    # -----------------------
    # Plot loop
    # -----------------------
    for i, metric in enumerate(metrics):
        plot_results_ax(fig, axes[i], results_plot, "α", metric, params=True)
    create_legend(fig, axes)
    save_fig(core, file_name)


def plot_compatibility_diff_analysis(root_dir, core, k):
    results = read_results_file(root_dir, core, k)
    results = add_method_type(results)

    mask_2sided = np.logical_and(results["method"].str.contains("2sided"), np.logical_not(results["method"].str.contains("optim")))
    mask_2sided = np.logical_and(mask_2sided, np.logical_not(results["method_type"] == "fusion_rank"))
    mask_2sided = np.logical_and(mask_2sided, np.logical_not(results["method"].str.contains("ATT")))
    mask_2sided = np.logical_and(mask_2sided, np.logical_not(results["method_type"] == "fusion_rank_score"))

    mask_keep = np.logical_and(np.logical_not(mask_2sided), np.logical_not(results["method"] == "UserJob"))
    mask_fair_optim = np.logical_and(results["method"].str.contains("2sided"), results["method"].str.contains("fair_fusion_optim"))
    results_fair_optim = results[mask_fair_optim]
    methods_fair_fusion_optim = ["2sided_fair_fusion_optim__DGI__country", "2sided_fair_fusion_optim__DGI__is_payed",
     "2sided_fair_fusion_optim__DUP__country"]
    results_fair_optim = results_fair_optim[results_fair_optim["method"].apply(lambda x: x in methods_fair_fusion_optim)]
    
    
    results = get_best_params(results, core, k)
    results = pd.concat([results[mask_keep], results_fair_optim])
    results = format_naming(results, group_flag=None)


    # --- Define metrics to plot ---
    metrics = [
        "compatibility_added_0","compatibility_added_1",
        "compatibility_added_de","compatibility_added_non-de",
        "compatibility_removed_0","compatibility_removed_1",
        "compatibility_removed_de","compatibility_removed_non-de"
    ]

    # --- Melt into long format ---
    plot_df = results.melt(
        id_vars=["method"],
        value_vars=metrics,
        var_name="metric", value_name="value"
    )

    plot_df["type"] = plot_df["metric"].apply(lambda x: "added" if "added" in x else "removed")
    plot_df["attribute"] = (
        plot_df["metric"]
        .str.replace("compatibility_added_", "")
        .str.replace("compatibility_removed_", "")
    )

    dict_map_gr = {
        "1": "Premium",
        "0": "Non-premium",
        "de": "German",
        "non-de": "Non-German"
    }
    plot_df["attribute"] = plot_df["attribute"].apply(lambda x: dict_map_gr[str(x)])

    # --- Add method-specific % of added items ---
    perc_cols = {
        "German": "p_added_country_de",
        "Non-German": "p_added_country_non-de",
        "Premium": "p_added_is_payed_1",
        "Non-premium": "p_added_is_payed_0"
    }

    perc_df = results[["method"] + list(perc_cols.values())].copy()
    perc_df = perc_df.melt(
        id_vars=["method"], 
        var_name="perc_col", 
        value_name="perc_value"
    )

    perc_df["attribute"] = perc_df["perc_col"].map({
        "p_added_country_de": "German",
        "p_added_country_non-de": "Non-German",
        "p_added_is_payed_1": "Premium",
        "p_added_is_payed_0": "Non-premium"
    })

    custom_order = [
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "userfairness(country)", "userfairness(premium)",
        "PCT", "CP-Fair(country)", "CP-Fair(premium)",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "TSF", "TSF - ATT", 
        
    ]

    # merge method + attribute
    plot_df = plot_df.merge(perc_df[["method", "attribute", "perc_value"]],
                            on=["method", "attribute"], how="left")

    plot_df["method"] = pd.Categorical(
    plot_df["method"],
    categories=custom_order,
    ordered=True
    )

    # Sort dataframe according to the custom order
    plot_df = plot_df.sort_values("method")


    attribute_order = ['Non-German', 'German', 'Non-premium', 'Premium']
    method_order = [
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "userfairness(country)", "userfairness(premium)",
        "PCT", "CP-Fair(country)", "CP-Fair(premium)",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "TSF", "TSF - ATT"
    ]
    type_colors = {"added": "#2E86AB", "removed": "#F26457"}
    n_methods = len(method_order)
    width = 0.8 / n_methods  # total width of bar cluster per attribute

    # Ensure order
    plot_df["attribute"] = pd.Categorical(plot_df["attribute"], categories=attribute_order, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)

    
    from matplotlib.patches import Rectangle

    # --- Prepare the data ---
    removed_data = plot_df[plot_df["type"] == "removed"]
    added_data = plot_df[plot_df["type"] == "added"]

    # --- Plot manually to ensure each bar is separate ---
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.despine(ax=ax)
    attribute_order = ['Non-German', 'German', 'Non-premium', 'Premium']

    # We'll manually offset bars so that we have per-bar control
    methods = list(removed_data["method"].cat.categories)
    bar_width = 0.8 / len(methods)  # distribute across group

    sns.barplot(
        data=added_data,
        x="attribute", y="value",
        hue="method",
        order=attribute_order,
        dodge=True,
        alpha=0.5,
        palette=custom_palette_bar_plot,
        edgecolor="black",
        ax=ax,
    )

    hatch_pattern = ".."

    # For each attribute-method pair, compare and plot lower on top
    for i_attr, attr in enumerate(attribute_order):
        for j_meth, method in enumerate(methods):
            add_row = added_data[
                (added_data["attribute"] == attr) &
                (added_data["method"] == method)
            ]
            rem_row = removed_data[
                (removed_data["attribute"] == attr) &
                (removed_data["method"] == method)
            ]

            if add_row.empty and rem_row.empty:
                continue

            # Get values (default to 0 if missing)
            add_val = add_row["value"].values[0] if not add_row.empty else 0
            rem_val = rem_row["value"].values[0] if not rem_row.empty else 0
            color = custom_palette_bar_plot[method]

            # Compute x position (manual dodge)
            x = i_attr + j_meth * bar_width - 0.4 + bar_width / 2 - 0.03

            # Determine which bar goes on top
            lower_type = "added" if add_val < rem_val else "removed"
            top_first = (lower_type == "added")

            # Draw bars: higher one first (so lower goes on top)
            order = ["removed", "added"] #if top_first else ["added", "removed"]

            for t in order:
                val = rem_val #if t == "added" else rem_val
                #if t == "removed":
                rect = Rectangle(
                    (x, 0),
                    bar_width,
                    val,
                    facecolor=color,
                    alpha=0.5,
                    #edgecolor="black",
                    hatch=hatch_pattern,
                    linewidth=1.0,
                    zorder=3 #if t == lower_type else 2
                )
                
                ax.add_patch(rect)
    # --- Plot "added" on top using seaborn (lighter, no hatch) ---
    

    # Fix x-ticks
    ax.set_xticks(range(len(attribute_order)))
    ax.set_xticklabels(attribute_order)
    ax.set_xlabel("")
    ax.set_ylabel("Compatibility")

    # --- Legend ---
    type_patches = [
        mpatches.Patch(facecolor="gray", edgecolor="black", hatch=hatch_pattern,
                    label="Removed"),
        mpatches.Patch(facecolor="gray", edgecolor="black", alpha=0.8,
                    label="Added")
    ]
    ax.legend(handles=type_patches, title="Type", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()

    custom_order = [
        "TSF", "TSF - ATT", "PCT",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "CP-Fair(country)", "CP-Fair(premium)",
        "userfairness(country)", "userfairness(premium)"
    ]
    unique_methods = [m for m in custom_order if m in plot_df["method"].unique()]
    method_patches = [
        mpatches.Patch(color=custom_palette_bar_plot[m], label=m)
        for m in unique_methods
    ]
    fig.legend(
        handles=method_patches,
        title="Method",
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.17),
        title_fontsize=13  
    ) 

    plt.ylim(0.05, 0.4)
    plt.subplots_adjust(bottom=0.9)

    plt.tight_layout()
    file_name = f"analysis_core{core}_compatibility_overlap_K{k}.png"
    save_fig(core, file_name)


# --- Create a darker version of your palette for "removed" ---
def darken_color(color, factor=0.8):
    """Darken color by multiplying RGB values by `factor`."""
    rgb = mcolors.to_rgb(color)
    return tuple([c * factor for c in rgb])



def plot_compatibility_diff_analysis_(root_dir, core, k):
    results = read_results_file(root_dir, core, k)
    mask_keep = np.logical_not(results["method"].str.contains("fair_fusion"))
    mask_keep = np.logical_and(mask_keep, np.logical_not(results["method"] == "UserJob"))
    results = results[mask_keep]

    results = get_best_params(results, core, k)
    results = format_naming(results, group_flag=True)

    
    # --- Define metrics to plot ---
    metrics = [
        "compatibility_added_0","compatibility_added_1",
        "compatibility_added_de","compatibility_added_non-de",
        "compatibility_removed_0","compatibility_removed_1",
        "compatibility_removed_de","compatibility_removed_non-de"
    ]

    # --- Melt into long format ---
    plot_df = results.melt(
        id_vars=["method"],
        value_vars=metrics,
        var_name="metric", value_name="value"
    )

    plot_df["type"] = plot_df["metric"].apply(lambda x: "added" if "added" in x else "removed")
    plot_df["attribute"] = (
        plot_df["metric"]
        .str.replace("compatibility_added_", "")
        .str.replace("compatibility_removed_", "")
    )

    dict_map_gr = {
        "1": "Premium",
        "0": "Non-premium",
        "de": "German",
        "non-de": "Non-German"
    }
    plot_df["attribute"] = plot_df["attribute"].apply(lambda x: dict_map_gr[str(x)])

    # --- Add method-specific % of added items ---
    perc_cols = {
        "German": "p_added_country_de",
        "Non-German": "p_added_country_non-de",
        "Premium": "p_added_is_payed_1",
        "Non-premium": "p_added_is_payed_0"
    }

    perc_df = results[["method"] + list(perc_cols.values())].copy()
    perc_df = perc_df.melt(
        id_vars=["method"], 
        var_name="perc_col", 
        value_name="perc_value"
    )

    perc_df["attribute"] = perc_df["perc_col"].map({
        "p_added_country_de": "German",
        "p_added_country_non-de": "Non-German",
        "p_added_is_payed_1": "Premium",
        "p_added_is_payed_0": "Non-premium"
    })

    custom_order = [
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)",
        "userfairness(country)", "userfairness(premium)",
        "PCT", "CP-Fair(country)", "CP-Fair(premium)",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "TSF", "TSF - ATT", 
        
    ]

    # merge method + attribute
    plot_df = plot_df.merge(perc_df[["method", "attribute", "perc_value"]],
                            on=["method", "attribute"], how="left")

    plot_df["method"] = pd.Categorical(
    plot_df["method"],
    categories=custom_order,
    ordered=True
    )

    # Sort dataframe according to the custom order
    plot_df = plot_df.sort_values("method")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(5, 6))
    plt.subplots_adjust(bottom=0.25)

    attribute_order = ['Non-German', 'German', 'Non-premium', 'Premium']
    # --- Plot added ---
    added_data = plot_df[plot_df["type"] == "added"]
    

    
    darker_palette = {k: darken_color(v, 1) for k, v in custom_palette_bar_plot.items()}

    # --- Plot removed (overlay) ---
    removed_plot = sns.barplot(
        data=plot_df[plot_df["type"] == "removed"],
        x="attribute", y="value",
        hue="method",
        order=attribute_order,
        dodge=True,
        alpha=0.6,
        edgecolor=None,
        palette=darker_palette
    )

    hatch_pattern = "."  # choose from '/', '\\', 'x', etc.
    new_patches = []
    from matplotlib.patches import Rectangle

    hatch_pattern = "."  # choose from "/", "\\", "x", "o", "-", etc.
    for bar in removed_plot.patches:
        bar.set_hatch(hatch_pattern)
        # force the hatch to be redrawn independently
        bar.set_zorder(2)
        bar.set_facecolor(bar.get_facecolor())  # trigger internal redraw

    added_plot = sns.barplot(
    data=plot_df[plot_df["type"] == "added"],
    x="attribute", y="value",
    hue="method",
    order=attribute_order,
    dodge=True,
    alpha=0.85,
    edgecolor="black",
    linewidth=1.2,
    zorder=2,
    palette=custom_palette_bar_plot
    )

    plt.xlabel("")
    plt.ylabel("Compatibility")

    # --- Legends ---
    ax = plt.gca()
    type_patches = [
        mpatches.Patch(facecolor="gray", edgecolor="black", linewidth=1.2,
                    alpha=0.8, label="Added"),
        mpatches.Patch(facecolor="black", edgecolor="black", linewidth=1.2,
                    alpha=0.6, label="Removed (darker)")
    ]
    ax.legend(handles=type_patches, title="Type", loc="upper right", frameon=True)

    custom_order = [
        "TSF", "TSF - ATT",
        "CP-Fair(country)", "PCT", "CP-Fair(premium)",
        "userfairness(country)", "userfairness(premium)",
        "TFROM(country)", "TFROM(premium)", "TFROM(country,premium)",
        "FA*IR(country)", "FA*IR(premium)", "FA*IR(country,premium)"
    ]
    unique_methods = [m for m in custom_order if m in plot_df["method"].unique()]
    method_patches = [
        mpatches.Patch(color=custom_palette_bar_plot[m], label=m)
        for m in unique_methods
    ]
    fig.legend(
        handles=method_patches,
        title="Method",
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.1)
    )
    plt.subplots_adjust(bottom=0.2)

    # --- Save figure ---
    file_name = f"analysis_core{core}_compatibility_overlap_K{k}.png"
    save_fig(core, file_name)






root_dir = "/home/crus/XINGInteractions/DATA"


#generate_results_file(root_dir, "20", k=10)


plot_metric_vs_alpha(root_dir, "20", k=10, rank_fusion=False)
#plot_metric_vs_alpha(root_dir, "20", k=10, rank_fusion=False)
# plot_compatibility_added_removed_items_analysis(root_dir, "20", k=10)
# plot_compatibility_diff_analysis(root_dir, "20", k=20)

