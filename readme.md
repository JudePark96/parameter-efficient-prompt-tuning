# Parameter-efficient Few-shot Learning based on Prompting

### 1. Requirements

```bash
torch==1.10.1
transformers=4.21.2
sentencepiece==0.1.97
scikit-learn==1.1.2
datasets==2.4.0
```

### 2. Preprocessing

#### Preprocessing few-shot learning data

First, download the raw dataset for building the **k-shot** few-shot dataset.

```bash
bash script/download_dataset.sh
python3 data_util/generate_k_shot_data.py --k k \
                                          --data_dir data_dir \
                                          --output_dir output_dir
```

Now you can build the features of few-shot dataset for each tasks.

```
bash script/preprocessing/preprocessing_finetuning_data.sh
```

### 3. Training

```bash
bash script/few_shot.prompting.training.sh
bash script/conventional_tuning.full_params.training.sh
bash script/conventional_tuning.freeze.training.sh
```

### 4. Aggregate Results

```bash
bash script/aggregate_results.sh
```

### Contact
```
judepark@jbnu.ac.kr
```
