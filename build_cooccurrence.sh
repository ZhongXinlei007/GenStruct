#!/bin/bash
# 从训练集构建共现关系PMI

DATASET_NAME="msnbc"  # 修改为你的数据集名称
TRAIN_FILE="datasets/${DATASET_NAME}/${DATASET_NAME}_train_co.jsonl"  # 修改为训练集路径
OUTPUT_DIR="cooccurrence"
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_pmi.json"
MIN_COOCCURRENCE=1  # 最小共现次数

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Error: training file not found at ${TRAIN_FILE}"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}

python -m new_code.build_cooccurrence \
    --train_file ${TRAIN_FILE} \
    --output_file ${OUTPUT_FILE} \
    --dataset_name ${DATASET_NAME} \
    --min_cooccurrence ${MIN_COOCCURRENCE}

echo "Co-occurrence PMI built: ${OUTPUT_FILE}"

