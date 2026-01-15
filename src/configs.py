import os
import argparse
root_path = "./DATA"

def read_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--method", type=str, help="2sided, UserJob, TFROM")
    parser.add_argument("--core", type=str, help="20, 50")
    parser.add_argument("--k", type=int, help="top-k")

    parser.add_argument("--N", type=int, help="size of input list")

    parser.add_argument("--alpha", type=str, help="alpha value for 2sided")
    parser.add_argument("--groups", type=str, help="group for [TFROM, CFP, IFTR, FairStar]")
    parser.add_argument("--param", type=str, help="param for [CFP, FairStar, PCT]")

    args = parser.parse_args()
    return args 

def get_configs(method = None, alpha = None, groups=None, k=None, core=None, param=None, N=None, args_bool=True):
    if args_bool:
        args = read_args()
        if method is None:
            method = args.method

        if alpha is None:
            alpha = args.alpha
        
        if groups is None:
            groups = args.groups

        if k is None:
            k = args.k

        if N is None:
            N = args.N

        if core is None:
            core = args.core
        
        if param is None:
            param = args.param
    
    if method in ["FairStar", "CFP"]:
        p = float(param.split(",")[0].replace("(", ""))
        a = float(param.split(",")[1].replace(")", ""))
    elif method in ["PCT"]:
        a = param.split(",")[0].replace("(", "")
        p = float(param.split(",")[1].replace(")", ""))
    else:
        a = None
        p = None
    

    dict_data_path = {
        "2sided_core20": os.path.join(root_path, "2SidedJobRec", f"2sided_userjob_binary_core20_10Krandomusers_1_N64603_alpha{alpha}_K100"),
        "2sided_core50": os.path.join(root_path, "2SidedJobRec", f"2sided_userjob_binary_core50_N2547_alpha{alpha}_K100"),

        "UserJob_core20": os.path.join(root_path, "Baseline_BPR", f"UserJob_binary_core20_10Krandomusers_1_K500"),
        "UserJob_core50": os.path.join(root_path, "Baseline_BPR", f"UserJob_binary_core50_K500"),
        
        "TFROM_core20": os.path.join(root_path, "Baseline_TFROM", f"UserJob_TFROM_uniform_{groups}_core20_10Krandomusers_1_N{N}_K50"),
        "TFROM_core50": os.path.join(root_path, "Baseline_TFROM", f"UserJob_TFROM_uniform_{groups}_core50_N{N}_K50"),
        
        "CFP_core50": os.path.join(root_path, "Baseline_CFP", f"UserJob_CFP_CP_{groups}_core50_N200_K100_ueps{a}_ieps{p}"),
        "CFP_core20": os.path.join(root_path, "Baseline_CFP", f"UserJob_CFP_CP_{groups}_core20_10Krandomusers_1_N200_K100_ueps{a}_ieps{p}"),
        
        "FairStar_core50": os.path.join(root_path, "Baseline_FairStar", f"UserJob_FairStar_{groups}_core50_p{p}_a{a}_N{N}_K100"),               
        "FairStar_core20": os.path.join(root_path, "Baseline_FairStar", f"UserJob_FairStar_{groups}_core20_10Krandomusers_1_p{p}_a{a}_N{N}_K100"), 
        
        "userfairness_core50": os.path.join(root_path, "Baseline_userfairness", f"UserJob_userfairness_{param}@{k}_{groups}_core50_N{N}_K100"),               
        "userfairness_core20": os.path.join(root_path, "Baseline_userfairness", f"UserJob_userfairness_{param}@{k}_{groups}_core20_N{N}_K100"), 
        
        "ITFR_core50": os.path.join(root_path, "Baseline_ITFR", f"UserJob_ITFR_{groups}_core50_K100"),
        "ITFR_core20": os.path.join(root_path, "Baseline_ITFR", f"UserJob_ITFR_{groups}_core20_10Krandomusers_1_K100"),
        
        "PCT_core50": os.path.join(root_path, "Baseline_PCT", f"UserJob_PCT_core50_N2547_K100_lmbd{param}"),
        "PCT_core20": os.path.join(root_path, "Baseline_PCT", f"UserJob_PCT_core20_10Krandomusers_1_N200_K100_{a}_lmbd{p}_{groups}"),
        
        "2sided_ATT_core20": os.path.join(root_path, "core-20", "2SidedJobRec", f"2sided_userjob_binary_core20_10Krandomusers_1_N64603_ATT_K100"),
        "2sided_ATT_core50": os.path.join(root_path, "core-50", "2SidedJobRec", f"2sided_userjob_binary_core50_N2547_ATT_K100"),
    }

    dict_experiment_data = {
    
        "path_user_job_split": os.path.join(root_path, "splits_userjob", f"core_{core}"),
        "recommendation_dir": os.path.join(root_path, f"recommendation/core-{core}"),
        "out_path": os.path.join(root_path, f"fairness_evaluation/core-{core}"),
        "k": k
        
    }

    
    data_exp = method + "_core" + core
    dict_experiment_data["data_path"] = dict_data_path[data_exp]

    if N is not None:
        data_exp += f"_N{N}_"

    if method == "2sided":
        data_exp += f"_alpha_{alpha}" 
    if method in ["TFROM", "CFP", "ITFR", "FairStar", "userfairness", "PCT"]:
        data_exp += f"_groups_{groups}" 
    
    if method in ["CFP", "FairStar", "PCT", "userfairness"]:
        data_exp += f"_param_{param}" 

    if method == "userfairness":
        data_exp += f"@{k}"
         
    data_exp += f"_k_{k}" 

    dict_experiment_data["save_path"] = os.path.join(root_path, "fairness_evaluation", data_exp)
    if not os.path.exists(dict_experiment_data["save_path"]):
        os.makedirs(dict_experiment_data["save_path"])
    return dict_experiment_data


