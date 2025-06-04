#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}

export GPUNUM=8
export MASTER_PORT=16899

export MODEL="1b"

export TGT="gemma"

MODEL_NAME="./data/pythia2${TGT}/${MODEL}/glove"

# export DATASET_PATH="./data/pretrain-dataset/pile00-${TGT}-tokenized"
export DATASET_PATH="./data/pretrain-dataset/pile00-sample-${TGT}-tokenized"

export CONFIG_FILE="./data/Deepspeed-Configs/zero3.yaml"

export TRAIN_BS=8
export EVAL_BS=1
export GRADIENT_ACC=16

export BLOCK_SIZE=2048

export SEED=0

export LR=6.4e-4
export NUM_STEPS=2500
export NUM_SAVE_STEPS=2500
export EVAL_STEP=10000
export NUM_WORKERS=0
export LOGGING_STEPS=1

export RESUME=False

export TRAIN_START_IDX=0

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S1"

if [ "${RESUME}" != "False" ];
then
PREFIX="${PREFIX}_resume"
ADD_PARAMETERS="${ADD_PARAMETERS} --resume_from_checkpoint ${RESUME}"
fi

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR


accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --finetune_embed_only True \
    --use_flash_attn True 2>&1 >$LOG_FILE

# STAGE-2
MODEL_NAME="./$MODEL_DIR/checkpoint-$NUM_STEPS"
LR=5e-5
export TRAIN_START_IDX=2560000

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${SEED}_${TGT}_S2"

MODEL_DIR="${MAIN_DIR}/log/$PREFIX"
LOG_FILE="${MAIN_DIR}/log/${PREFIX}.log"

mkdir -p $MODEL_DIR

accelerate launch \
    --config_file ${CONFIG_FILE} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${GPUNUM} \
    --num_machines 1 src/clm_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --ignore_data_skip True \
    --train_start_idx ${TRAIN_START_IDX} \
    ${ADD_PARAMETERS} \
    --warmup_ratio 0.03 \
    --use_flash_attn True 2>&1 >$LOG_FILE
  
