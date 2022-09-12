#!/bin/bash

bash script/training/few_shot.conventional_tuning.freeze.training.sh 2
bash script/training/few_shot.conventional_tuning.full_params.training.sh 2
bash script/training/few_shot.prompting.training.sh 2
bash script/training/few_shot.prefix+prompting.training.sh 2
