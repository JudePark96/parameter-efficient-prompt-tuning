#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

pre_seq_len=15
pre_hidden_size=512

model_name_or_config_path='roberta-base'
pretrained_prefix_path="../checkpoints/PrefixPreTraining/${model_name_or_config_path^^}_PREFIX_HIDDEN${pre_hidden_size}_PREFIX_LEN${pre_seq_len}_E5_B64_WARM0.06_NORM1.0_LR3e-05/4epoch.pth"

for task in "SST-2" "cr" "RTE" "mr"; do
  for seed in "13" "21" "42" "87" "100"; do
    input_path="../rsc/preprocessed/finetuning/${model_name_or_config_path}/${task}/16-${seed}"
    checkpoint_path="../checkpoints/finetuning/${model_name_or_config_path}/prefix+prompting/pre_seq_len${pre_seq_len}/${task}/16-${seed}"
    python3 fewshot_trainer.py --num_train_epochs 50 \
                           --model_name_or_config_path ${model_name_or_config_path} \
                           --model_type 'prefix+prompt' \
                           --pretrained_prefix_path ${pretrained_prefix_path} \
                           --prefix_hidden_size 512 \
                           --pre_seq_len ${pre_seq_len} \
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