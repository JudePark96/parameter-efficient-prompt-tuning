#!/bin/bash

for task in "SST-2" "cr" "mr" ; do
  for seed in "13" "21" "42" "87" "100"; do
    input_path="../rsc/k-shot/${task}/16-${seed}"
    output_path="../rsc/preprocessed/finetuning/${task}/16-${seed}"

    python3 data_util/create_features.py --input_path ${input_path} \
                                         --output_path ${output_path} \
                                         --task ${task} \
                                         --preprocessing_type "finetune" \
                                         --max_length 128 \
                                         --seed ${seed}
  done
done

task="RTE"
for seed in "13" "21" "42" "87" "100"; do
  input_path="../rsc/k-shot/${task}/16-${seed}"
  output_path="../rsc/preprocessed/finetuning/${task}/16-${seed}"

  python3 data_util/create_features.py --input_path ${input_path} \
                                           --output_path ${output_path} \
                                           --task ${task} \
                                           --preprocessing_type "finetune" \
                                           --max_length 256 \
                                           --seed ${seed}
done