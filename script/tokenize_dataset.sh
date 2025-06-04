#!/bin/sh

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}
export CACHE_DIR="${MAIN_DIR}/data/cache"

export MODLE_PATH="./data/pythia2gemma/TokAlign-Init-1B"
export TOKENIZER_PATH="./data/pythia2gemma/TokAlign-Init-1B"

# export TRAIN_FILE="./data/pretrain-corpus/pile00.json"
export TRAIN_FILE="./data/pretrain-corpus/pile00.sample.json"

# export DATASET_PATH="./data/pretrain-dataset/pile00-gemma-tokenized"
export DATASET_PATH="./data/pretrain-dataset/pile00-sample-gemma-tokenized"

export NUM_WORKERS=60
export BLOCK_SIZE=2048

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -u src/process_dataset.py \
  --model_name_or_path ${MODLE_PATH} \
  --tokenizer_name ${TOKENIZER_PATH} \
  --train_file ${TRAIN_FILE} \
  --cache_dir ${CACHE_DIR} \
  --dataset_path_in_disk ${DATASET_PATH} \
  --preprocessing_num_workers ${NUM_WORKERS} \
  --block_size ${BLOCK_SIZE} \
  --output_dir ./log 2>&1 | tee ./log/process_dataset.log