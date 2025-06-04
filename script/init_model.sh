#!/bin/sh

export MAIN_DIR="/path/2/TokAlign/"
cd ${MAIN_DIR}

# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2gemma/align_matrix.json"
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2gemma/align_matrix_demo.json"

export MODLE_PATH1="EleutherAI/pythia-1b"

export TOKENIZER_PATH2="google/gemma-2b"

export OUTPUT_PATH="${MAIN_DIR}/data/pythia2gemma/TokAlign-Init-1B"

python src/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}
