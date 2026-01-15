import os
import pandas as pd

# =========================================
# Directory handling functions
# =========================================

def get_eval_dirs(method, root_dir, fusion_type, core, k):
    """
    Get directories for evaluation for a given method, core, fusion_type, and k.
    """
    if fusion_type is None:
        dirs_ = [
            d for d in os.listdir(root_dir)
            if f"core{core}" in d and method in d and f"_k_{k}" in d
        ]
        if method == "2sided":
            dirs_ = [d for d in dirs_ if "ATT" not in d]
    else:
        dirs_ = [
            d for d in os.listdir(root_dir)
            if f"core{core}" in d and "2sided" in d
            and d.endswith(f"K{k}") and f"{fusion_type}_K" in d
        ]
        if not dirs_:
            dirs_ = [
                d for d in os.listdir(root_dir)
                if f"core{core}" in d and "2sided" in d
                and d.endswith(f"K{k}") and f"{fusion_type}" in d
            ]
        if fusion_type == "isr":
            dirs_ = [d for d in dirs_ if "log_" not in d]
    return dirs_


def get_recsys_dirs(method, root_dir, fusion_type, core, k):
    """
    Get directories for recsys output.
    """
    if fusion_type is None:
        if method == "2sided":
            dirs_path = os.path.join(root_dir, "2SidedJobRec")
            dirs_ = [
                os.path.join(dirs_path, d)
                for d in os.listdir(dirs_path)
                if f"core{core}" in d and method in d and f"_K{k}" in d
            ]
        elif method == "UserJob":
            dirs_path = os.path.join(root_dir, "Baseline_BPR")
            dirs_ = [
                os.path.join(dirs_path, d)
                for d in os.listdir(dirs_path)
                if f"core{core}" in d and "UserJob" in d
            ]
        elif method == "2sided_ATT":
            dirs_path = os.path.join(root_dir, f"core-{core}", "2SidedJobRec")
            dirs_ = [
                os.path.join(dirs_path, d)
                for d in os.listdir(dirs_path)
                if f"core{core}" in d and "ATT" in d
            ]
        else:
            dirs_path = os.path.join(root_dir, f"Baseline_{method}")
            dirs_ = [
                os.path.join(dirs_path, d)
                for d in os.listdir(dirs_path)
                if f"core{core}" in d and method in d
            ]
    else:
        dirs_path = root_dir
        dirs_ = [
            os.path.join(dirs_path, d)
            for d in os.listdir(dirs_path)
            if f"core{core}" in d and "2sided" in d and f"{fusion_type}_K" in d
        ]
        if not dirs_:
            dirs_ = [
                os.path.join(dirs_path, d)
                for d in os.listdir(dirs_path)
                if f"core{core}" in d and "2sided" in d and f"{fusion_type}" in d
            ]
        if fusion_type == "isr":
            dirs_ = [os.path.join(dirs_path, d) for d in dirs_ if "log_" not in d]
    return dirs_


def get_params(dir_, fusion_type):
    """
    Extract parameter(s) from the directory name.
    """
    split_key = []
    if "alpha" in dir_:
        split_key.append("alpha")
    if "param" in dir_:
        split_key.append("param")
    if "N" in dir_ and fusion_type is None:
        split_key.append("N")
    if not split_key:
        return None

    params = []
    for key in split_key:
        if fusion_type is not None:
            params.append(dir_.split(key)[1].split("_")[0])
        else:
            if key == "N":
                params.append(dir_.split(key)[1].split("_")[0])
            else:
                params.append(dir_.split(f"{key}_")[1].split("_")[0])
    if len(params) > 1:
        return f"({','.join(params)})"
    return params[0]


def get_groups(dir_):
    """
    Extract group information from directory name.
    """
    if "groups" in dir_:
        groups = dir_.split("groups_")[1].split("_")[0]
        if "is_payed" in dir_:
            groups = groups.replace("is", "ispayed")
        return groups
    return None


def get_eval_dir(root_dir, dir_, fusion_type):
    """
    Return path to evaluation directory.
    """
    if fusion_type is None:
        return os.path.join(root_dir, dir_)
    return os.path.join(root_dir, dir_, "fairness_evaluation")


def get_recsys_file(root_dir, dir_, fusion_type):
    """
    Return path to recsys output file.
    """
    _file = os.path.join(root_dir, dir_, "out.txt")
    if not os.path.exists(_file):
        _file = os.path.join(root_dir, dir_, "out-1.txt")
    return _file


def get_method_name(method, groups, fusion_type):
    """
    Construct method name with group/fusion_type if applicable.
    """
    if groups is not None:
        return f"{method}({groups})"
    if fusion_type is not None:
        return f"{method}_{fusion_type}"
    return method


# =========================================
# Reading metrics
# =========================================

def read_metric(eval_dir, metric, group, per_group):
    """
    Read group fairness or diversity metric from files.
    """
    if metric in dict_metrics_group_fairness:
        if group == "__country__is_payed":
            file_name = dict_metrics_group_fairness[metric]["proportional_population"] + "_per_group_items.csv"
            path = os.path.join(eval_dir,
                                dict_metrics_group_fairness[metric]["proportional_data"],
                                group + "_no_discrepancy",
                                dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" + file_name)
            return pd.read_csv(path)['eval'].mean()

        elif group == "intersectional__country__is_payed":
            file_name = dict_metrics_group_fairness[metric]["proportional_population"] + ".csv_per_group_items.csv"
            path = os.path.join(eval_dir,
                                dict_metrics_group_fairness[metric]["proportional_data"],
                                group.replace("intersectional", "") + "_no_discrepancy",
                                "intersectional_" + dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" + file_name)
            df = pd.read_csv(path)
            df = df.groupby("__country__is_payed")["eval"].mean().reset_index()
            return df.rename(columns={df.columns[-1]: metric})

        elif per_group:
            file_name = dict_metrics_group_fairness[metric]["proportional_population"] + "_per_group_items.csv"
            path = os.path.join(eval_dir,
                                dict_metrics_group_fairness[metric]["proportional_data"],
                                group + "_no_discrepancy",
                                dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" + file_name)
            df = pd.read_csv(path)
            if not per_UID:
                df = df.groupby(group)["eval"].mean().reset_index()
                df = df.rename(columns={df.columns[-1]: metric})
            return df
        else:
            file_name = dict_metrics_group_fairness[metric]["proportional_population"] + ".csv"
            path = os.path.join(eval_dir,
                                dict_metrics_group_fairness[metric]["proportional_data"],
                                group + "_no_discrepancy",
                                dict_metrics_group_fairness[metric]["measure"] + "__" + "proportional_" + file_name)
            return pd.read_csv(path)['0'].values[0]

    elif metric in dict_metrics_diversity:
        path = os.path.join(eval_dir, dict_metrics_diversity[metric], metric + "_all.csv")
        return pd.read_csv(path)[metric].values[0]


def read_compatibility(eval_dir, metric):
    """
    Read compatibility metric from appropriate file.
    """
    compatibility_analysis = False

    if metric == "compatibility_score":
        path = os.path.join(root_dir, eval_dir, "compatibility_wc", "compatibility_user_job.csv")
    elif metric in ["compatibility_added_de", "compatibility_added_non-de",
                    "compatibility_removed_de", "compatibility_removed_non-de"]:
        metric_file = metric
        if "removed" in metric:
            metric_group = metric.split("_")[-1]
            metric_file = metric.replace(metric_group, f"_{metric_group}")
        path = os.path.join(root_dir, eval_dir, "country_rec_analysis_wc", f"{metric_file}.csv")
    elif metric in ["compatibility_added_0", "compatibility_added_1",
                    "compatibility_removed_0", "compatibility_removed_1"]:
        metric_file = metric
        if "removed" in metric:
            metric_group = metric.split("_")[-1]
            metric_file = metric.replace(metric_group, f"_{metric_group}")
        path = os.path.join(root_dir, eval_dir, "is_payed_rec_analysis_wc", f"{metric_file}.csv")
    elif "__" in metric:
        path = os.path.join(root_dir, eval_dir, "compatibility", "compatibility_user_job.csv")
        compatibility_analysis = True

    if not os.path.exists(path):
        return -1

    df = pd.read_csv(path)
    if not compatibility_analysis:
        return df["match_score"]
    else:
        group = metric.split("__")[1]
        value = metric.split("__")[2]
        raw = df.loc[0, group].split()
        pairs = [(raw[3], float(raw[4])), (raw[6], float(raw[7]))]
        return pd.DataFrame(pairs, columns=["group_name","value"]).query("group_name == @value")["value"]


# =========================================
# Utility / Diversity metric reader
# =========================================

def read_utility_diversity_metrics(method, fusion_type, core, metric, k, user_group, item_group, params):
    """
    Reads utility/diversity metrics for a method and parameter configuration.
    """
    columns_map = {
        "bpr": ['attr','precision','recall','ndcg','coverage','gini'],
        "TFROM": ['attr','criteria','feature','N','precision','recall','ndcg','coverage','gini'],
        "CFP": ['attr','feature','N','epsilon','precision','recall','ndcg','coverage','gini'],
        "2sided_ATT": ['attr','precision','recall','ndcg','coverage','gini'],
        "2sided": ['attr','alpha','precision','recall','ndcg','coverage','gini'],
        "weighted_sum": ['attr','alpha','precision','recall','ndcg','coverage','gini'],
        "userfairness": ['attr','feature','N','precision','recall','ndcg','coverage','gini'],
        "FairStar": ['attr','feature','N','p','a','precision','recall','ndcg','coverage','gini']
    }

    dir_path = os.path.join(root_dir, "performance_evaluation")
    files = os.listdir(dir_path)
    if method == "UserJob": method = "bpr"

    file_match = [f for f in files if f"core{core}" in f and f"K{k}" in f]

    if fusion_type is not None and fusion_type != "weighted_sum":
        if "fair_fusion_optim" in fusion_type:
            file_match = [f for f in file_match if "fairfusion" in f]
        else:
            file_match = [f for f in file_match if "rankfusion" in f]
    elif "ATT" in method:
        file_match = [f for f in file_match if "ATT" in f]
    else:
        file_match = [f for f in file_match if method.lower() in f and "ATT" not in f]

    if len(file_match) > 1:
        raise Exception("My ERROR: too many files")
    if not file_match:
        return -1

    results = pd.read_csv(os.path.join(dir_path, file_match[0]), sep="\t", header=None)

    if fusion_type and fusion_type != "weighted_sum":
        if "fair_fusion_optim" in fusion_type:
            columns = ['attr','fusion_type','precision','recall','ndcg','coverage','gini']
        else:
            columns = ['attr','fusion_type','alpha','precision','recall','ndcg','coverage','gini']
    else:
        columns = columns_map[method]

    results.columns = columns
    results = results[results["attr"] == user_group]

    # Map features
    if "feature" in columns:
        if item_group == "country": item_group = "c"
        elif item_group in ["premium","ispayed"]: item_group = "p"
        elif item_group in ["countryispayed","countrypremium"]: item_group = "cp"
        results = results[results['feature'] == item_group]

    # alpha / epsilon / fusion_type filters
    if "alpha" in columns:
        if "fusion_type" in columns and ("min" in fusion_type or "max" in fusion_type):
            params = 0
        results = results[results['alpha'].astype(float) == float(params)]
    if "epsilon" in columns:
        results = results[results['epsilon'].astype(float) == float(params)]

    # FairStar specific
    if method == "FairStar":
        p_params = params.split(",")[0].replace("((", "")
        a_params = params.split(",")[1].replace(")", "")
        N_params = params.split(",")[2].replace(")", "")
        results = results[results['p'].astype(float) == float(p_params)]
        results = results[results['a'].astype(float) == float(a_params)]

    # N filter
    if "N" in columns:
        if method == "CFP": N = "200"
        elif method == "userfairness": N = params.split(",")[1].split(")")[0]
        elif method == "FairStar": N = N_params
        else: N = params
        results = results[results['N'].astype(int) == int(N)]

    # fusion_type filter
    if "fusion_type" in columns:
        if "fair_fusion_optim" in fusion_type:
            results["fusion_type"] = "fair_fusion_optim__" + results["fusion_type"].astype(str)
        if "min" in fusion_type or "max" in fusion_type:
            results["fusion_type"] = "comb_" + results["fusion_type"].astype(str)
        results = results[results["fusion_type"] == fusion_type]

    if len(results) > 1:
        print(results)
        raise Exception("MY ERROR: too many results")
    if len(results) == 0:
        return -1
    return results[metric].values[0]


# =========================================
# Evaluation file readers
# =========================================

def read_evaluation_file(method, root_dir, core, group, metric, per_group=False, per_UID=False, fusion_type=None, k=10):
    """
    Reads evaluation files for a single method, returns a DataFrame of metric values.
    """
    dirs_eval = get_eval_dirs(method, root_dir, fusion_type, core, k)
    metric_values = []

    for dir_eval in dirs_eval:
        params = get_params(dir_eval, fusion_type)
        groups = get_groups(dir_eval)

        # Utility metrics
        if metric in ['precision','recall','ndcg','coverage','gini']:
            for user_group in ["All","German","NonGerman","Premium","NonPremium"]:
                metric_val = read_utility_diversity_metrics(method, fusion_type, core, metric, k, user_group, groups, params)
                method_name = get_method_name(method, groups, fusion_type)
                df_metric = pd.DataFrame({
                    "method":[method_name],
                    "params":[params],
                    "groups":[groups],
                    f"{metric}({user_group})":[float(metric_val)]
                })
                metric_values.append(df_metric)

        # Compatibility metrics
        elif "compatibility" in metric:
            eval_dir = get_eval_dir(root_dir, dir_eval, fusion_type)
            if fusion_type: eval_dir = eval_dir.replace("fairness_evaluation","")
            metric_val = read_compatibility(eval_dir, metric)

        # Group fairness
        else:
            eval_dir = get_eval_dir(root_dir, dir_eval, fusion_type)
            metric_val = read_metric(eval_dir, metric, group, per_group)

        # Store non-utility metrics
        if metric not in ['precision','recall','ndcg','coverage','gini']:
            method_name = get_method_name(method, groups, fusion_type)
            if not "intersectional" in group and not per_group:
                df_metric = pd.DataFrame({
                    "method":[method_name],
                    "params":[params],
                    "groups":[groups],
                    metric:[float(metric_val)]
                })
            else:
                df_metric = metric_val.copy()
                df_metric["method"] = [method]*len(df_metric)
                df_metric["params"] = [params]*len(df_metric)
                df_metric["groups"] = [groups]*len(df_metric)

            metric_values.append(df_metric)

    if not metric_values:
        return None
    return pd.concat(metric_values, ignore_index=True)


def format_metric_dataframe(df, metric, utility_metrics, group_name):
    """
    Helper: formats metric DataFrame into long format with metric column.
    """
    formatted = []

    if metric in utility_metrics:
        for user_group in ["All","German","NonGerman","Premium","NonPremium"]:
            col_name = f"{metric}({user_group})"
            temp_df = df[["method","params","groups",col_name]].copy()
            temp_df = temp_df.rename(columns={col_name:"value"})
            temp_df["metric"] = col_name
            formatted.append(temp_df[["method","params","groups","metric","value"]])
    elif "compatibility" in metric or "p_" in metric:
        df = df.rename(columns={metric:"value"})
        df["metric"] = metric
