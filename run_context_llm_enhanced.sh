#!/bin/bash

################################################################################
# LLM增强版Context推理脚本 - 通用版
# 用法: 修改 DATASET_NAME 变量后执行此脚本
################################################################################

# ============================================================================
# 配置区域 - 根据需要修改以下参数
# ============================================================================

# 数据集名称 (可选项: ace2004, aida, aquaint, cweb, msnbc, wiki)
DATASET_NAME="ace2004"

# LLM模型路径
LLM_MODEL_PATH="/data2/home/E24301323/code/llm/zephyr-7b-beta"

# GPU设置
CUDA_DEVICE=0

# ============================================================================
# 以下代码通常不需要修改
# ============================================================================

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "=========================================="
echo "LLM增强版Context推理 - 数据集: $DATASET_NAME"
echo "=========================================="
echo "数据集名称: $DATASET_NAME"
echo "LLM模型: $LLM_MODEL_PATH"
echo "GPU设备: $CUDA_DEVICE"
echo "=========================================="
echo ""

# 定义文件路径
INPUT_FILE="datasets/${DATASET_NAME}/listwise_filter_llm_enhanced.jsonl"
OUTPUT_FILE="result/${DATASET_NAME}_context_llm_enhanced.json"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    echo "请先运行 bash run_llm_enhanced_context.sh 生成LLM增强上下文"
    exit 1
fi

echo "步骤2: 使用LLM增强上下文进行Context推理"
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 执行Context推理（使用LLM增强版本）
# 注意: 需要修改 new_code/prompt.py 以支持LLM增强上下文
python3 -u new_code/prompt_enhanced.py \
  --dataset_name $DATASET_NAME \
  --dataset_path $INPUT_FILE \
  --model_name Zephyr \
  --model_path $LLM_MODEL_PATH \
  --func_name context \
  --exp_name ${DATASET_NAME}_context_llm_enhanced

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Context推理完成！"
    echo "=========================================="
    echo ""
    echo "生成文件:"
    echo "  - $OUTPUT_FILE"
    echo ""
    echo "下一步: 处理结果并评估"
    echo "python3 process_llm_enhanced_result.py --dataset_name $DATASET_NAME"
else
    echo ""
    echo "=========================================="
    echo "❌ 推理失败！请检查错误信息"
    echo "=========================================="
    exit 1
fi