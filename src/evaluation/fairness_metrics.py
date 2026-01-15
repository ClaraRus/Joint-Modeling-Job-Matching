import itertools
import os
import time
import numpy as np
import pandas as pd
import math

from scipy.spatial.distance import jensenshannon


def compute_fairness_group(row, metric):
    if "count" in row:
        if row[metric] == 0 and row["count"] == 0:
            return 1
        elif row[metric] == 0:
            return 0
        elif row["count"] == 0:
            return row[metric] / 0.01
        else:
            return row[metric] / row["count"]
    else:
        return row[metric]

def compute_fairness_intersectional(x, metric):
    if "count" in x:
        js = jensenshannon(x[metric], x["count"])
    else:
        js = jensenshannon(x[metric], [0.25, 0.25, 0.25, 0.25])
    
    return js

def compute_fairness_metric_difference(x, unique_groups, group_col):
    diff = []   
    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            diff.append(x[x[group_col] == unique_groups[i]]['eval'].values[0] -
                        x[x[group_col] == unique_groups[j]]['eval'].values[0])

    return max(diff)

def compute_fairness_metric_per_group(x, unique_groups, group_col):
    results = {}
    for unique_group in unique_groups:
        results[unique_group] = x[x[group_col] == unique_group]['eval'].values[0]
    return results

def compute_metric(metric, save_path, data, query_col, group_col, k, data_proportional=None,
                   per_ID_prop=False, global_population=False):

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    file_name = metric
    if data_proportional is not None:
        if per_ID_prop:
            file_name = file_name + "__proportional_clicks_item_groups_per_query(preference)" # DUP - user preference
        else:
            file_name = file_name + "__proportional_population_groups_items" # DGI - item fairness

        if global_population:
            file_name = file_name + "__global" # DGI(global) - global item fairness 
    
    file_name = file_name + ".csv"
    

    if not os.path.exists(file_name):
        unique_groups = data[group_col].unique()
      
        if metric == "diversity":
            res = compute_diversity(data, query_col, group_col, k, global_population)
            
        elif metric == "exposer":
            res = compute_exposer(data, query_col, group_col, k, global_population)

        if not global_population:
            all_combinations = pd.MultiIndex.from_product([res[query_col].unique(), res[group_col].unique()],
                                                      names=[query_col, group_col])
            res = res.set_index([query_col, group_col]).reindex(all_combinations, fill_value=0).reset_index()

        if data_proportional is not None:
            # set group_column for data_proportional
            # based on which groups we count the proportionality
           
            if per_ID_prop:
                # proportional to the clicks on the item group per query
                group_columns = [query_col, group_col]
            else:
                # proportional to the population of item groups
                group_columns = [group_col]

            count_groups = data_proportional.groupby(group_columns).apply(lambda x: len(x)).reset_index()
            count_groups = count_groups.rename(columns={0: "count"})

            # normalize the count of the groups
            if per_ID_prop:
                # per query
                count_groups["count"] = count_groups[[query_col, "count"]].apply(
                    lambda x: x["count"] / count_groups[count_groups[query_col] == x[query_col]]["count"].sum(), axis=1)
                count_groups = count_groups.set_index([query_col, group_col]).reindex(all_combinations,
                                                                                      fill_value=0).reset_index()
                res = pd.merge(count_groups, res, on=[query_col, group_col])
            else:
                # per population in the whole data
                count_groups["count"] = count_groups["count"].apply(lambda x: x / count_groups["count"].sum())
                count_group_column = group_col

                res["count"] = res[group_col].apply(
                    lambda x: count_groups[count_groups[count_group_column] == unique_groups[0]]['count'].values[
                        0] if x == unique_groups[0] else
                    count_groups[count_groups[count_group_column] == unique_groups[1]]['count'].values[0])
        
        if query_col not in res:
                res[query_col] = [list(data[query_col].unique())[0]] * len(res)

        if len(unique_groups) > 2:
            res_js = res.groupby(query_col).apply(lambda x: compute_fairness_intersectional(x, metric)).reset_index()
            res_js = res_js.rename(columns={res.columns[-1]: 'eval'})
            fairness_per_query = res_js

            average_fairness = np.mean(np.abs(fairness_per_query))
            fairness_per_query = fairness_per_query.reset_index()
            fairness_per_query = pd.concat(
                [fairness_per_query, pd.DataFrame([{query_col: "average", 0: average_fairness}])], ignore_index=True)

            fairness_per_query_average = fairness_per_query[fairness_per_query[query_col] == "average"]
            
            fairness_per_query_average.to_csv(os.path.join(save_path, file_name))

        res['eval'] = res.apply(lambda x: compute_fairness_group(x, metric), axis=1)
        res = res[res[query_col].isin(res[query_col].unique())]
       
        if len(unique_groups) <= 2:
            fairness_per_query = res.groupby(query_col).apply(
            lambda x: compute_fairness_metric_difference(x, unique_groups, group_col))

            average_fairness = np.mean(np.abs(fairness_per_query))
            fairness_per_query = fairness_per_query.reset_index()
            fairness_per_query = pd.concat(
            [fairness_per_query, pd.DataFrame([{query_col: "average", 0: average_fairness}])], ignore_index=True)

            fairness_per_query_average = fairness_per_query[fairness_per_query[query_col] == "average"]
            
            
            fairness_per_query_per_group = res.groupby(query_col).apply(
            lambda x: compute_fairness_metric_per_group(x, unique_groups, group_col))
            fairness_df = pd.DataFrame(fairness_per_query_per_group.tolist(),
                           index=fairness_per_query_per_group.index)
            avg_per_group = fairness_df.mean(axis=0).reset_index()
            # Rename columns
            avg_per_group.columns = ["group", "eval"]

            if save_path is not None:
                if not os.path.exists(os.path.join(save_path, file_name)):
                    fairness_per_query_average.to_csv(os.path.join(save_path, file_name))

        return fairness_per_query


def compute_diversity(data, query_col, group_col, k, global_population):
    if global_population:
        columns_to_group = [group_col]
    else:
        columns_to_group = [query_col, group_col]
    res = data.groupby(columns_to_group).apply(lambda x: len(x)).reset_index()
    res = res.rename(columns={0: "diversity"})
    if global_population:
        res["diversity"] = res["diversity"] / len(data)
    else:
        res["diversity"] = res["diversity"] / k

    return res


def compute_exposer(data, query_col, group_col, k, global_population):
    if global_population:
        columns_to_group = [group_col]
    else:
        columns_to_group = [query_col, group_col]
    res = data.groupby(columns_to_group).apply(lambda x: sum(1 / np.log(x["rank"] + 1))).reset_index()
    res = res.rename(columns={0: "exposer"})
    if global_population:
        res["exposer"] = res["exposer"] / len(data)
    else:
        res["exposer"] = res["exposer"] / k

    return res


def compute_metrics(data_dict, k, save_path, per_ID_prop=False, global_population=False):
    # per_ID_group - compute fairness proportional to (preference of the user) the clicks on the groups of items per query

    configs = list(data_dict.keys())
    if per_ID_prop:
        configs_proportional = [{"proportional_user_preference": data_dict["proportional_user_preference"]}]
    else:
        configs_proportional = [{"proportional_item_population": data_dict["proportional_item_population"]}]
   

    data_proportional = [{"non_proportional": None}]
    data_proportional.extend(configs_proportional)

    permutations = itertools.product(data_dict["groups"], data_proportional)
    for permutation in permutations:
        group, proportional = permutation
        if save_path is not None:
            save_path_metric = os.path.join(save_path, list(proportional.keys())[0], group)
        else:
            save_path_metric = None
        

        res_diversity = compute_metric("diversity", save_path_metric, data_dict["recommendation"],
                                       data_dict["query_col"],
                                       group, k,
                                       data_proportional=list(proportional.values())[0], 
                                       per_ID_prop=per_ID_prop, global_population=global_population)

        res_exposure = compute_metric("exposer", save_path_metric, data_dict["recommendation"], data_dict["query_col"],
                                      group, k,
                                      data_proportional=list(proportional.values())[0], 
                                      per_ID_prop=per_ID_prop, global_population=global_population)

    return res_diversity, res_exposure
