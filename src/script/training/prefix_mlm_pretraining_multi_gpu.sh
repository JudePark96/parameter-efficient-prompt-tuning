#!/bin/bash
CORES=$(lscpu | grep Core | awk '{print $4}')
SOCKETS=$(lscpu | grep Socket | awk '{print $2}')
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=48
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=$(expr $RANDOM + 1000)
export MASTER_ADDR="localhost"

DEVICE_IDS=0,1,2,3

NUM_TRAIN_EPOCHS=5
PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2

PEAK_LR=3e-05
#WARMUP_PROPORTION=0.3
WARMUP_PROPORTION=0.06
MAX_GRAD_NORM=1.0

BATCH_SIZE=`expr ${PER_GPU_TRAIN_BATCH_SIZE} \* ${GRADIENT_ACCUMULATION_STEPS} \* ${N_GPU_NODE}`

# Model Setting
DEV_EVAL_STEP=500
OPTIMIZER='adafactor'

# HuggingFace Config Setting
MODEL_NAME_OR_CONFIG_PATH='roberta-large'
INPUT_PATH='../rsc/preprocessed/pretraining/'
FILE_NAME="1.5m_${MODEL_NAME_OR_CONFIG_PATH}_maxlen384_prob0.15_max_pred_per_seq20_do_whole_word_maskTrue"
PREFIX_HIDDEN_SIZE=512

for PRE_SEQ_LEN in 10 15 20; do
  SAVE_CHECKPOINTS_DIR="../checkpoints/PrefixPreTraining/${MODEL_NAME_OR_CONFIG_PATH^^}_WWM_TRUE_PREFIX_HIDDEN${PREFIX_HIDDEN_SIZE}_PREFIX_LEN${PRE_SEQ_LEN}_E${NUM_TRAIN_EPOCHS}_B${BATCH_SIZE}_WARM${WARMUP_PROPORTION}_NORM${MAX_GRAD_NORM}_LR${PEAK_LR}"
  python3 -m torch.distributed.launch \
            --nproc_per_node ${N_GPU_NODE} \
            --nnodes ${N_NODES} \
            --node_rank ${NODE_RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            pretraining_trainer.py --n_gpu ${WORLD_SIZE} \
                    --device_ids ${DEVICE_IDS} \
                    --model_name_or_config_path ${MODEL_NAME_OR_CONFIG_PATH} \
                    --save_checkpoints_dir "${SAVE_CHECKPOINTS_DIR}" \
                    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
                    --input_path ${INPUT_PATH} \
                    --file_name ${FILE_NAME} \
                    --pre_seq_len ${PRE_SEQ_LEN} \
                    --prefix_hidden_size ${PREFIX_HIDDEN_SIZE} \
                    --dev_eval_step ${DEV_EVAL_STEP} \
                    --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
                    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
                    --learning_rate ${PEAK_LR} \
                    --optimizer ${OPTIMIZER} \
                    --max_grad_norm ${MAX_GRAD_NORM} \
                    --warmup_proportion ${WARMUP_PROPORTION}
done