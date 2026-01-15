# =========================
# Fusion types
# =========================
FUSION_TYPES = [
    "fair_fusion_optim__DGI__country",
    "fair_fusion_optim__DUP__country",
    "fair_fusion_optim__DGI(global)__country",
    "fair_fusion_optim__DGI__is_payed",
    "fair_fusion_optim__DUP__is_payed",
    "fair_fusion_optim__DGI(global)__is_payed",
]

# =========================
# User groups
# =========================
USER_GROUPS = ["All", "German", "NonGerman", "Premium", "NonPremium"]

# =========================
# Utility metrics
# =========================
UTILITY_METRICS = ["precision", "recall", "ndcg", "coverage", "gini"]

UTILITY_METRIC_LABELS = {
    "precision": "Precision@K",
    "recall": "Recall@K",
    "ndcg": "nDCG@K",
    "coverage": "Coverage",
    "gini": "Gini Index",
}

# =========================
# Group fairness metrics
# =========================
GROUP_FAIRNESS_METRICS = {
    "DGI": {"measure": "DGI", "proportional_data": "proportional", "proportional_population": "items"},
    "EGI": {"measure": "EGI", "proportional_data": "proportional", "proportional_population": "items"},
    "DUP": {"measure": "DUP", "proportional_data": "proportional", "proportional_population": "users"},
    "EUP": {"measure": "EUP", "proportional_data": "proportional", "proportional_population": "users"},
}

GROUP_FAIRNESS_METRIC_LABELS = {
    "DGI": "DGI ↓",
    "EGI": "EGI ↓",
    "DUP": "DUP ↓",
    "EUP": "EUP ↓",
}

GLOBAL_GROUP_FAIRNESS_METRICS = {
    "DGI(global)": {"measure": "DGI", "proportional_data": "global_proportional", "proportional_population": "items"},
    "EGI(global)": {"measure": "EGI", "proportional_data": "global_proportional", "proportional_population": "items"},
}

GLOBAL_GROUP_FAIRNESS_METRIC_LABELS = {
    "DGI(global)": "DGI (global) ↓",
    "EGI(global)": "EGI (global) ↓",
}

ALL_FAIRNESS_METRICS = list(GROUP_FAIRNESS_METRICS.keys()) + list(GLOBAL_GROUP_FAIRNESS_METRICS.keys())
ALL_METRICS = UTILITY_METRICS + ALL_FAIRNESS_METRICS

# =========================
# Method color palettes
# =========================
METHOD_PALETTE = {
    "BPR": "red",
    "TFROM(country)": "blue",
    "TFROM(premium)": "orange",
    "TFROM(country,premium)": "green",
    "CFP(country)": "blue",
    "CFP(premium)": "orange",
    "FA*IR(country)": "blue",
    "FA*IR(premium)": "orange",
    "FA*IR(country,premium)": "green",
    "userfairness(country)": "blue",
    "userfairness(premium)": "orange",
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
    "TSF - Fair(RGI(premium))": "orange",
    "TSF - Fair(RGI(country,premium))": "green",
    "TSF - Fair(RUP(country))": "blue",
    "TSF - Fair(RUP(premium))": "brown",
}

# =========================
# Bar plot colors
# =========================
BAR_PLOT_PALETTE = {
    "BPR": "#e41a1c",
    "TFROM(country)": "#1f77b4",
    "TFROM(premium)": "#6baed6",
    "TFROM(country,premium)": "#08306b",
    "FA*IR(country)": "#2ca02c",
    "FA*IR(premium)": "#98df8a",
    "FA*IR(country,premium)": "#145214",
    "CP-Fair(country)": "#ffbf00",
    "CP-Fair(premium)": "#ffe680",
    "userfairness(country)": "#8c564b",
    "userfairness(premium)": "#c49c94",
    "TSF": "#800080",
    "TSF - ATT": "#ff69b4",
}

# =========================
# Helper functions
# =========================
def is_utility_metric(metric: str) -> bool:
    return metric in UTILITY_METRICS

def is_fairness_metric(metric: str) -> bool:
    return metric in ALL_FAIRNESS_METRICS

def get_metric_label(metric: str) -> str:
    """
    Return a human-readable label for any metric.
    """
    if metric in UTILITY_METRIC_LABELS:
        return UTILITY_METRIC_LABELS[metric]
    if metric in GROUP_FAIRNESS_METRIC_LABELS:
        return GROUP_FAIRNESS_METRIC_LABELS[metric]
    if metric in GLOBAL_GROUP_FAIRNESS_METRIC_LABELS:
        return GLOBAL_GROUP_FAIRNESS_METRIC_LABELS[metric]
    return metric

def get_group_fairness_config(metric: str) -> dict:
    """
    Returns the configuration needed to locate group-fairness files.
    """
    if metric in GROUP_FAIRNESS_METRICS:
        return GROUP_FAIRNESS_METRICS[metric]
    if metric in GLOBAL_GROUP_FAIRNESS_METRICS:
        return GLOBAL_GROUP_FAIRNESS_METRICS[metric]
    raise KeyError(f"Unknown group fairness metric: {metric}")


def format_naming(results):
    """
    Standardize method names in the results DataFrame.
    """
    name_map = {
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
        "2sided_fair_fusion_optim__DGI__country":"TSF - Fair(RGI(country))",
        "2sided_fair_fusion_optim__DGI__is_payed":"TSF - Fair(RGI(premium))",
        "2sided_fair_fusion_optim__DUP__country":"TSF - Fair(RUP(country))",
        "2sided_fair_fusion_optim__DUP__is_payed":"TSF - Fair(RUP(premium))",
    }
    results["method"] = results["method"].apply(lambda x: name_map.get(x, x))
    return results


def add_method_type(results):
    """
    Add a column describing method type: item, fusion_score, two-sided, etc.
    """
    type_map = {
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
        'TFROM(country)': "two-sided",
        'TFROM(ispayed)': "two-sided",
        'TFROM(countryispayed)': "two-sided",
        "2sided_fair_fusion_optim__DGI__country":"fusion_learn",
        "2sided_fair_fusion_optim__DGI__is_payed":"fusion_learn",
        "2sided_fair_fusion_optim__DUP__country":"fusion_learn",
        "2sided_fair_fusion_optim__DUP__is_payed":"fusion_learn",
        "2sided_fair_fusion_optim__DGI(global)__country":"fusion_learn",
        "2sided_fair_fusion_optim__DGI(global)__is_payed":"fusion_learn",
    }
    results["method_type"] = results["method"].apply(lambda x: type_map.get(x, x))
    return results


def get_best_params(results, core, k=10):
    """
    Select the subset of results corresponding to the "best" parameter configuration
    based on core and k.
    """
    import numpy as np
    params_num = pd.to_numeric(results["params"], errors="coerce")
    
    mask = pd.Series(False, index=results.index)

    if core == "20" and k == 10:
        best_params_map = {
            "2sided": 0.5,
            "2sided_isr": 0.5,
            "2sided_log_isr": 0.5,
            "2sided_borda_fuse": 0.5,
            "2sided_rrf": 0.5,
            "CFP(country)": 0.5,
            "CFP(premium)": 0.5,
            "PCT": 0.4,
            "FairStar(ispayed)": "((0.6, 0.15),1000)",
            "FairStar(country)": "((0.6, 0.1),1000)",
            "FairStar(countryispayed)": "((0.6, 0.05),1000)",
            "userfairness(country)": f"(precision@{k},200)",
            "userfairness(premium)": f"(precision@{k},200)",
            "TFROM(country)": 200,
            "TFROM(ispayed)": 200,
            "TFROM(countryispayed)": 200
        }
        for method_name, param_value in best_params_map.items():
            mask |= np.logical_and(results["method"] == method_name,
                                   (results["params"] == param_value) | (params_num == param_value))
    
    # Always include rows with missing or empty params
    mask |= results["params"].isna()
    mask |= results["params"].isin(["None", ""])
    
    return results[mask]
