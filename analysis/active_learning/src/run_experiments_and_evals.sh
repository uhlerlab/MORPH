#!/bin/bash

# Define the list of leave_out_test_set_id and method values
random_seed=(12)
dataset_name="replogle_k562_essential_hvg"
representation_type="DepMap_GeneEffect"
leave_out_test_set_ids=("active_learning_fold_1")
device="cuda:4"

methods=("baseline")
# methods=("baseline" "learn_loss_mse" "active_learning_kmeans_center")
use_prior="False"
prior_only="False"
cluster_first="False"

num_round=5
initial_num_ptb=100
num_ptb=100
batch_size=32
initial_epochs=100
incremental_epochs=100
decay_factor=1 # set to 1 if learn_from_scratch
tolerance_epochs=20
val_ratio=0.1

# Eval specific parameters
n_top_deg=50

# Loop over each combination of leave_out_test_set_id and method
for leave_out_test_set_id in "${leave_out_test_set_ids[@]}"; do
    for method in "${methods[@]}"; do
        for seed in "${random_seed[@]}"; do
            echo "Running experiment with leave_out_test_set_id=$leave_out_test_set_id and method=$method and seed=$seed and use_groundtruth_matrix=$use_groundtruth_matrix"
            python run.py --random_seed "$seed" \
                            --dataset_name "$dataset_name" \
                            --representation_type "$representation_type" \
                            --leave_out_test_set_id "$leave_out_test_set_id" \
                            --device "$device" \
                            --method "$method" \
                            --use_prior "$use_prior" \
                            --prior_only "$prior_only" \
                            --cluster_first "$cluster_first" \
                            --num_round "$num_round" \
                            --initial_num_ptb "$initial_num_ptb" \
                            --num_ptb "$num_ptb" \
                            --batch_size "$batch_size" \
                            --initial_epochs "$initial_epochs" \
                            --incremental_epochs "$incremental_epochs" \
                            --decay_factor "$decay_factor" \
                            --tolerance_epochs "$tolerance_epochs" \
                            --val_ratio "$val_ratio"

            echo "Running evaluations with leave_out_test_set_id=$leave_out_test_set_id, method=$method, seed=$seed, use_groundtruth=$use_groundtruth_matrix"
            python evaluate.py --random_seed "$seed" \
                                --dataset_name "$dataset_name" \
                                --representation_type "$representation_type" \
                                --leave_out_test_set_id "$leave_out_test_set_id" \
                                --method "$method" \
                                --device "$device" \
                                --use_prior "$use_prior" \
                                --prior_only "$prior_only" \
                                --cluster_first "$cluster_first" \
                                --n_top_deg "$n_top_deg" \
                                --num_round "$num_round" \
                                --initial_num_ptb "$initial_num_ptb" \
                                --num_ptb "$num_ptb" \
                                --batch_size "$batch_size" \
                                --initial_epochs "$initial_epochs" \
                                --incremental_epochs "$incremental_epochs" \
                                --val_ratio "$val_ratio" \
                                --decay_factor "$decay_factor" \
                                --tolerance_epochs "$tolerance_epochs"
        done
    done
done