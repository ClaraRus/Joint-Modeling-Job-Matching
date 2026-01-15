from src.utils import process_data, add_intersectional_group_column
from src.configs import get_configs, read_args
import itertools
import os
import time
import numpy as np
import pandas as pd
import argparse

def candidate_job_compatibility(x, users, items, fields):
    dict_compatibility = dict()
    if x['J_ID'] not in items['id'].values:
        return None
    if x['U_ID'] not in users['id'].values:
        return None
    for field in fields:
        dict_compatibility[field] = users[users['id'] == x['U_ID']][field].values[0] == \
                                    items[items['id'] == x['J_ID']][field].values[0]
        dict_compatibility['U_ID'] = x['U_ID']
        dict_compatibility['J_ID'] = x['J_ID']
        dict_compatibility['item_country'] = x['country']
        dict_compatibility['item_premium'] = x['is_payed']
        dict_compatibility['user_country'] = users[users['id'] == x['U_ID']]["country"].values[0]
        dict_compatibility['user_premium'] = users[users['id'] == x['U_ID']]["premium"].values[0]
    return dict_compatibility

def compatibility_score(recommendation, data_users, data_items, path, file_name, group=None, fields=['industry_id', 'discipline_id', 'career_level', 'country']):
    if len(recommendation) > 0 :
        users = data_users[data_users['id'].isin(recommendation['U_ID'])]
        items = data_items[data_items['id'].isin(recommendation['J_ID'])]
    

        compatibility = recommendation.apply(lambda x: candidate_job_compatibility(x, users, items, fields), axis=1)
        compatibility = pd.DataFrame(list(compatibility[compatibility != None].values))
        compatibility["match_score"] = (compatibility[fields].sum(axis=1)) / len(fields)
    
        df_mean = pd.DataFrame.from_dict({
            "match_score": [compatibility["match_score"].mean()],
        })
    else:
        df_mean = pd.DataFrame.from_dict({
            "match_score": [None]
        })

    
    df_mean.to_csv(os.path.join(path, file_name))



def compatibility_analysis(recommendation_1sided, recommendation_2sided, data_users, data_items, group, group_1, group_2, save_path_, fields=['industry_id', 'discipline_id', 'career_level', 'country']):
    save_path = os.path.join(save_path_, group + "_compatibility_analysis")
    os.makedirs(save_path, exist_ok=True)

    for run in range(1, 2):
        recommendation_user_job_set = set(map(tuple, recommendation_1sided[['U_ID', 'J_ID']].values))
        check_in_recommendation = lambda x: tuple(x) in recommendation_user_job_set
        recommendation_2sided['confirmed'] = recommendation_2sided[['U_ID', 'J_ID']].apply(
            check_in_recommendation, axis=1)
        mask_2sided_added = recommendation_2sided['confirmed'] == False

        print("ADDED ITEMS", sum(mask_2sided_added))

        recommendation_2sided_job_set = set(map(tuple, recommendation_2sided[['U_ID', 'J_ID']].values))
        check_in_recommendation = lambda x: tuple(x) in recommendation_2sided_job_set
        recommendation_1sided['confirmed'] = recommendation_1sided[['U_ID', 'J_ID']].apply(
            check_in_recommendation, axis=1)
        mask_1sided_removed = recommendation_1sided['confirmed'] == False
        
        print("REMOVED ITEMS", sum(mask_1sided_removed))

        # new group_1 recommended in 2sided
        added_items = recommendation_2sided[mask_2sided_added]
        premium_added_items = added_items[added_items[group] == group_1]
        compatibility_score(premium_added_items, data_users, data_items, save_path,
                    file_name="compatibility_added_" + str(group_1) + ".csv", fields=fields)

        # new group_1 recommended in 2 sided
        added_items = recommendation_2sided[mask_2sided_added]
        non_premium_added_items = added_items[added_items[group] == group_2]
        compatibility_score(non_premium_added_items, data_users, data_items, save_path,
                    file_name="compatibility_added_" + str(group_2) + ".csv", fields=fields)

        # group_1 not recommended anymore in the 2sided (which were recommended before in the 1 sided)
        removed_items = recommendation_1sided[mask_1sided_removed]
        premium_removed_items = removed_items[removed_items[group] == group_1]
        compatibility_score(premium_removed_items, data_users, data_items, save_path,
                    file_name="compatibility_removed__" + str(group_1) + ".csv", fields=fields)

        # group_2 not recommended anymore in the 2sided (which were recommended before in the 1 sided)
        removed_items = recommendation_1sided[mask_1sided_removed]
        premium_removed_items = removed_items[removed_items[group] == group_2]
        compatibility_score(premium_removed_items, data_users, data_items, save_path,
                    file_name="compatibility_removed__" + str(group_2) + ".csv", fields=fields)
