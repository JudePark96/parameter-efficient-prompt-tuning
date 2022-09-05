#!/bin/bash

echo "aggregating prompting results"
for task in "SST-2" "cr" "RTE" "mr"; do
  python3 aggregate_few_shot_results.py "../checkpoints/finetuning/prompting/${task}/" "${task}" "16" "13,21,42,87,100"
done

echo "aggregating conventional_freeze results"
for task in "SST-2" "cr" "RTE" "mr"; do
  python3 aggregate_few_shot_results.py "../checkpoints/finetuning/conventional_freeze/${task}/" "${task}" "16" "13,21,42,87,100"
done

echo "aggregating conventional_full_params results"
for task in "SST-2" "cr" "RTE" "mr"; do
  python3 aggregate_few_shot_results.py "../checkpoints/finetuning/conventional_full_params/${task}/" "${task}" "16" "13,21,42,87,100"
done