#!/bin/bash

input_path=../rsc/wikipedia/eng_wiki.txt
output_path=../rsc/preprocessed/pretraining
model_name_or_config_path='roberta-large'

python3 data_util/create_features.py --input_path ${input_path} \
                                     --output_path ${output_path} \
                                     --preprocessing_type 'pretraining' \
                                     --model_name_or_config_path ${model_name_or_config_path} \
                                     --max_length 384 \
                                     --do_whole_word_mask

python3 data_util/create_features.py --input_path ${input_path} \
                                     --output_path ${output_path} \
                                     --preprocessing_type 'pretraining' \
                                     --model_name_or_config_path ${model_name_or_config_path} \
                                     --max_length 384