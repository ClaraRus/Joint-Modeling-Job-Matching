# Joint-Modeling-of-Candidate-and-Recruiter-Preferences-for-Fair-Two-Sided-Job-Matching

Recommender systems in recruitment platforms involve two active sides, candidates and recruiters, each with distinct goals and preferences. Most recommendation methods address only one side of the problem, leading to potentially ineffective matches. We propose a two-sided fusion framework that jointly models candidate and recruiter preferences to enhance mutual matches between candidates and recruiters. We also propose a personalized two-sided fusion approach to enhance the fairness of job recommendations. Experiments on the XING recruitment dataset show that the proposed approach improves fairness and compatibility, demonstrating the benefits of incorporating two-sided preferences in fairness-aware recommendations.


Please cite this work:

Rus, C., Mansoury, M., Yates, A., & de Rijke, M. (2026). Joint modeling of candidate and recruiter preferences for fair two-sided job matching. In Proceedings of the 48th European Conference on Information Retrieval (ECIR).

# Requirements
python 3.8

```
conda env create -f environment.yml
conda activate fairness_env
```
# Run

You can find the run scripts under ```/src/TSF/run_TSF.sh```.

### Running TSF
```
python run_two_sided_fusion.py --fusion_type <fusion_type> --core "20"
```

```fusion_type``` can take the following values:
- weighted_sum
- comb_max
- comb_min
- rrf
- isr
- borda_fuse

### Running TSF-Fair
```
python run_two_sided_fusion.py --fusion_type fair_fusion_optim --core "20" --settings <settings> --weight <w>
```
```settings``` can take the following predefined values:
- settings_DGI_country
- settings_DUP_country
- settings_DGI_is_payed
- settings_DGI_DUP_country (```optimization_obj = w * DGI(country) + (1-w) * DUP(country)```)
- settings_DGI_country_is_payed (```optimization_obj = w * DGI(country) + (1-w) * DGI(is_payed)```)

You can define your own settings in ```run_two_sided_fusion.py``` as it follows:
```
settings_<metric>_<attribute> = [
    {
         "fair_metrics":["metric"],  # list of fairness metrics used during optimization
         "group_cols": ["attribute"] # list of attributes used during optimization 
    }
]
```
Next define your setting in ```settings_choices``` dict:
```
settings_choices = {
    "settings_<metric>_<attribute>": settings_<metric>_<attribute>
}
```
  
```w``` is the weight of the metrics if dual optimization is used, meaning ```optimization_obj = w * metric_1 + (1-w) * metric_2```. Code suports only dual weighted optimization, but if you want to optimize for more metrics or attributes the weight will be set to 1. 


### Example - Use Case


Below you can see an example of how the JOB recommendation list chnages due TSF - SUM (alpha = 0.7) for U_ID = 596135 in top-10, which preference towards NON-DE jobs is 0.2:
##### BPR
2251808
1244790
1623767
447115
1303059
721917
783762 - DE (removed)
1728106
1918159
74639 - DE (added)


##### TSF - SUM
1728106
447115
2251808
1918159
1056604 - NON-DE (added)
1623767
1244790 - DE (added)
1303059
721917
1731132

TSF - SUM adds one more NON-DE job, respecting more the user preference and increasing the fairness as it increses the number of NON-DE jobs which are overall underrepresented.

