#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

for task in "SST-2" "cr" "RTE" "mr"; do
  for seed in "13" "21" "42" "87" "100"; do
    input_path="../rsc/preprocessed/finetuning/${task}/16-${seed}"
    checkpoint_path="../checkpoints/finetuning/prefix+prompting/${task}/16-${seed}"
    python3 fewshot_trainer.py --num_train_epochs 50 \
                           --model_name_or_config_path 'roberta-base' \
                           --model_type 'prefix+prompt' \
                           --prefix_hidden_size 512 \
                           --pre_seq_len 10 \
                           --input_path ${input_path} \
                           --save_checkpoints_dir ${checkpoint_path} \
                           --task_name ${task} \
                           --train_batch_size 8 \
                           --eval_batch_size 8 \
                           --num_workers 12 \
                           --learning_rate 3e-5 \
                           --warmup_proportion 0.03 \
                           --max_grad_norm 1.0 \
                           --early_stopping 10
  done
done