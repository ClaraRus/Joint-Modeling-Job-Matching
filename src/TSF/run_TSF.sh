
conda activate fairness_env
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "comb_min" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "comb_max" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "borda_fuse" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "rrf" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "isr" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "weighted_sum" --core "20" 

python ./src/TSF/run_two_sided_fusion.py --fusion_type "fair_fusion_optim" --settings "settings_DGI_country_is_payed" --core "20" --weight "0"
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "fair_fusion_optim" --settings "settings_DGI_country" --core "20" 
# python ./src/TSF/run_two_sided_fusion.py --fusion_type "fair_fusion_optim" --settings "settings_DUP_country" --core "20" 
#python ./src/TSF/run_two_sided_fusion.py --fusion_type "fair_fusion_optim" --settings "settings_DGI_is_payed" --core "20"  

# python ./src/TSF/run_two_sided_fusion.py --fusion_type "fair_fusion_optim" --settings "settings_DGI_DUP_country" --core "20" --weight "1"





