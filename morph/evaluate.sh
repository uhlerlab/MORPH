#!/bin/bash

# Running models -------------------------------------------------------------
random_seeds=(12)
modality="rna" # "rna" or "ops"
model="MORPH" # "MORPH" or "MORPH_no_residual1" or "MORPH_moe_3expert"
dataset_name="norman_k562_hvg"
leave_out_test_set_ids=("random_split_1_2")
representation_types=("DepMap_GeneEffect")
label_2=None
label_3=None
null_label='zeros' # 'gaussian_normalized' or 'gaussian' or 'zeros'
device="cuda:4"
run_name="" # Name of the run, if not specified, will use the most recent run (e.g., "DepMap_GeneEffect_MORPH_run1742322142" or "")

model_name_list=("best_model.pt" "best_model_val.pt")

# Running models and evaluations ----------------------------------------------
for test_set_name in "${leave_out_test_set_ids[@]}"
do
    for representation_type in "${representation_types[@]}"
    do
        for random_seed in "${random_seeds[@]}"
        do
            echo "Running evaluations with test_set_name=$test_set_name, dataset_name=$dataset_name, random_seed=$random_seed"
            for model_name in "${model_name_list[@]}"
            do
                python evaluate.py --modality "$modality" \
                                   --dataset_name "$dataset_name" \
                                   --leave_out_test_set_id "$test_set_name" \
                                   --label "$representation_type" \
                                   --null_label "$null_label" \
                                   --model_type "$model" \
                                   --model_name "$model_name" \
                                   --random_seed "$random_seed" \
                                   --device "$device"
            done
        done
    done
done