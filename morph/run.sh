#!/bin/bash

# Running models -------------------------------------------------------------
random_seeds=(12)
modality="rna" # "rna" or "ops"
model="MORPH_moe_3expert" # "MORPH" or "MORPH_no_residual1" or "MORPH_moe_3expert"
dataset_name="norman_k562_hvg"
leave_out_test_set_ids=("random_split_1_2")
representation_types=("DepMap_GeneEffect")
label_2="GenePT_v1" #"GenePT_v1", None
label_3="STRING" #"STRING", None
null_label='zeros' # 'gaussian_normalized' or 'gaussian' or 'zeros'
epochs=100
tolerance_epochs=20 #set to 20 by default
MMD_sigma=1500
mxAlpha=2 # set to 2 by default
mxBeta=2 # set to 2 by default
Gamma1=1 #hyperparameter for reconstruction loss (MMD), range 0-1
Gamma2=0 #hyperparameter for reconstruction loss (MSE), range 0-1
latdim_ctrl=50
latdim_ptb=50
geneset_num=50
geneset_dim=50
device="cuda:5"

model_name_list=("best_model.pt" "best_model_val.pt")

# Running models and evaluations ----------------------------------------------
for test_set_name in "${leave_out_test_set_ids[@]}"
do
    for representation_type in "${representation_types[@]}"
    do
        for random_seed in "${random_seeds[@]}"
        do
            echo "Running models with test_set_name=$test_set_name, dataset_name=$dataset_name, random_seed=$random_seed"
            python run.py --modality "$modality" \
                          --random_seed "$random_seed" \
                          --dataset_name "$dataset_name" \
                          --leave_out_test_set_id "$test_set_name" \
                          --device "$device" \
                          --model "$model" \
                          --label "$representation_type" \
                          --label_2 "$label_2" \
                          --label_3 "$label_3" \
                          --null_label "$null_label" \
                          --epochs "$epochs" \
                          --tolerance_epochs "$tolerance_epochs" \
                          --MMD_sigma "$MMD_sigma" \
                          --mxAlpha "$mxAlpha" \
                          --mxBeta "$mxBeta" \
                          --Gamma1 "$Gamma1" \
                          --Gamma2 "$Gamma2" \
                          --latdim_ctrl "$latdim_ctrl" \
                          --latdim_ptb "$latdim_ptb" \
                          --geneset_num "$geneset_num" \
                          --geneset_dim "$geneset_dim"

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