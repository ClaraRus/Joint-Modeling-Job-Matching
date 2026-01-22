
import os
import pandas as pd
from src.utils import add_sensitive_columns, add_rank_column
import matplotlib.pyplot as plt
import numpy as np

def plot_subplots_by_alpha(df, file_name):
    # Get sorted unique alphas
    alphas = sorted(df["alpha"].unique())
    n = len(alphas)

    # Choose rows/cols for subplots (square-ish layout)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)

    for ax, alpha in zip(axes.flatten(), alphas):

        # Filter data for this alpha
        df_a = df[df["alpha"] == alpha]

        # Scatter plot
        ax.scatter(
            df_a["proportion_data"],
            df_a["proportion_preference"],
            alpha=0.7
        )

        # Proportionality dotted line (y=x)
        ax.plot([0,1], [0,1], linestyle="dotted")

        # Labels and title
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("proportion_data")
        ax.set_ylabel("proportion_preference")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Remove any unused axes (if #alphas < rows*cols)
    for empty_ax in axes.flatten()[len(alphas):]:
        empty_ax.axis("off")

    plt.tight_layout()
    fig.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_use_cases_preference(group, group_value, save_path):
    # read items
    path_items = "/home/crus/XINGInteractions/DATA/items.csv"
    data_items = pd.read_csv(path_items)
    data_items['country'] = data_items['country'].apply(lambda x: 'non-de' if x != 'de' else x)


    BPR_path = "/home/crus/XINGInteractions/DATA/Baseline_BPR/UserJob_binary_core20_10Krandomusers_1_K500/out-1.txt"
    BPR = pd.read_csv(BPR_path, header=None, names=["U_ID", "J_ID", "S"])


    if "country" not in BPR:
        BPR = add_sensitive_columns(BPR, 'J_ID', data_items,
                                                        ['country', 'is_payed'])
    BPR = BPR.groupby('U_ID', group_keys=False).apply(lambda x: x.nlargest(10, 'S')).reset_index(drop=True)

    results = []
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        alpha = str(alpha)
        path = f"/home/crus/XINGInteractions/DATA/core-20/2SidedJobRec/2sided_userjob_binary_core20_N64603_weighted_sum_alpha{alpha}_K10/out.txt"
        data = pd.read_csv(path, header=None, names=["U_ID", "J_ID", "S"])

        if "country" not in data:
            data = add_sensitive_columns(data, 'J_ID', data_items,
                                                            ['country', 'is_payed'])
        
        # find users for which <group> jobs are recommended more often than in BPR
        BPR_group_value_counts= BPR.groupby("U_ID")[group].value_counts(normalize=True).reset_index(name='proportion')
        data_group_value_counts= data.groupby("U_ID")[group].value_counts(normalize=True).reset_index(name='proportion')

        print(BPR_group_value_counts)
        print(data_group_value_counts)

        BPR_group_value_counts = BPR_group_value_counts[BPR_group_value_counts[group] == group_value]
        data_group_value_counts = data_group_value_counts[data_group_value_counts[group] == group_value]

        print(BPR_group_value_counts)
        print(data_group_value_counts)
        
        merged = BPR_group_value_counts.merge(data_group_value_counts, on=["U_ID", "country"], suffixes=('_BPR', '_data'), how='inner', indicator=True)
        merged['proportion_diff'] = merged['proportion_data'] - merged['proportion_BPR']
        merged = merged[merged['proportion_diff'] > 0]

        print(merged)
        print(len(merged['U_ID'].unique())/len(data_group_value_counts['U_ID'].unique()))

        # find users who have higher preferences for non-de jobs 
        test_data = pd.read_csv("/home/crus/XINGInteractions/DATA/splits_userjob/core_20/test_interactions_userjob_binary.txt", sep='\t',header=None, names=["U_ID", "J_ID", "S"])
        print(test_data)

        test_data = add_sensitive_columns(test_data, 'J_ID', data_items,
                                                            ['country', 'is_payed'])
        print(test_data)

        test_user_group_value_counts= test_data.groupby("U_ID")[group].value_counts(normalize=True).reset_index(name='proportion_preference')
        test_user_group_value_counts = test_user_group_value_counts[test_user_group_value_counts[group] == group_value]
        test_user_group_value_counts = test_user_group_value_counts[test_user_group_value_counts['U_ID'].isin(merged['U_ID'].values)]
        print(test_user_group_value_counts)
        merged = merged.merge(test_user_group_value_counts, on=["U_ID", "country"], how='inner', suffixes=('_recommendation', '_preference'))[['U_ID', 'country', 'proportion_BPR', 'proportion_data', 'proportion_preference']] 

        print(merged)
        print(merged.columns)
        merged["alpha"] = alpha
        results.append(merged)

       
    results_df = pd.concat(results, ignore_index=True)
    print(results_df.columns)

    print("TH 0.05")
    results_df['fairness'] = results_df['proportion_data'] - results_df['proportion_preference']
    mask = np.logical_and(results_df['fairness'] >= 0, results_df['fairness'] <= 0.05)
    results_df = results_df[mask]
    print(results_df)
    results_df.to_csv("/home/crus/XINGInteractions/DATA/core-20/plots/use_case_preference_high_0.05.csv", index=False)

    results_df['fairness'] = results_df['proportion_data'] - results_df['proportion_preference']
    mask = np.logical_and(results_df['fairness'] <= 0, results_df['fairness'] >= -0.05)
    results_df = results_df[mask]
    print(results_df)
    results_df.to_csv("/home/crus/XINGInteractions/DATA/core-20/plots/use_case_preference_low_0.05.csv", index=False)


    print("TH 0.1")
    results_df['fairness'] = results_df['proportion_data'] - results_df['proportion_preference']
    mask = np.logical_and(results_df['fairness'] >= 0, results_df['fairness'] <= 0.1)
    results_df = results_df[mask]
    print(results_df)
    results_df.to_csv("/home/crus/XINGInteractions/DATA/core-20/plots/use_case_preference_high_0.1.csv", index=False)

    results_df['fairness'] = results_df['proportion_data'] - results_df['proportion_preference']
    mask = np.logical_and(results_df['fairness'] <= 0, results_df['fairness'] >= -0.01)
    results_df = results_df[mask]
    print(results_df)
    results_df.to_csv("/home/crus/XINGInteractions/DATA/core-20/plots/use_case_preference_low_0.1.csv", index=False)

    file_name = os.path.join(save_path, f"use_case_preference_{group}_{group_value}.png")
    plot_subplots_by_alpha(results_df, file_name)

save_path = "/home/crus/XINGInteractions/DATA/core-20/plots"
get_use_cases_preference(group='country', group_value='non-de', save_path=save_path)