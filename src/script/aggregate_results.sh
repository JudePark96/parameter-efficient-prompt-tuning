#!/bin/bash

echo "aggregating prompting results"
for base in "roberta-base" "roberta-large"; do
  for length in 10 15 20; do
    for task in "SST-2" "cr" "RTE" "mr"; do
      python3 aggregate_few_shot_results.py "../checkpoints/finetuning/${base}/prompting/pre_seq_len${length}/${task}/" "${task}" "16" "13,21,42,87,100"
    done
  done
done

echo "aggregating prefix+prompting results"
for base in "roberta-base" "roberta-large"; do
  for masking_mode in 'FALSE' 'TRUE'; do
    for length in 10 15 20; do
      for task in "SST-2" "cr" "RTE" "mr"; do
        python3 aggregate_few_shot_results.py "../checkpoints/finetuning/${base}/prefix+prompting/WWM_${masking_mode}/pre_seq_len${length}/${task}/" "${task}" "16" "13,21,42,87,100"
      done
    done
  done
done

echo "aggregating conventional_freeze results"
for base in "roberta-base" "roberta-large"; do
  for task in "SST-2" "cr" "RTE" "mr"; do
    python3 aggregate_few_shot_results.py "../checkpoints/finetuning/${base}/conventional_freeze/${task}/" "${task}" "16" "13,21,42,87,100"
  done
done

echo "aggregating conventional_full_params results"
for base in "roberta-base" "roberta-large"; do
  for task in "SST-2" "cr" "RTE" "mr"; do
    python3 aggregate_few_shot_results.py "../checkpoints/finetuning/${base}/conventional_full_params/${task}/" "${task}" "16" "13,21,42,87,100"
  done
done