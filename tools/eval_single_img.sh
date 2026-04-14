#!/usr/bin/env bash

# 默认参数
INPUT_IMG="/home/dzy/data/CAD/stru3d/test/03250.png"
OUTPUT_IMG="/home/dzy/data/CAD/stru3d_out/03250_out.png"
OUTPUT_DXF="/home/dzy/data/CAD/stru3d_out/03250_out.dxf"
CHECKPOINT="checkpoints/roomformer_stru3d_semantic_rich.pth"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_img)
            INPUT_IMG="$2"
            shift 2
            ;;
        --output_img)
            OUTPUT_IMG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 --input_img <输入图片路径> --output_img <输出图片路径> [--checkpoint <模型路径>]"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$INPUT_IMG" || -z "$OUTPUT_IMG" ]]; then
    echo "缺少必需参数！"
    echo "使用方法: $0 --input_img <输入图片路径> --output_img <输出图片路径> [--checkpoint <模型路径>]"
    exit 1
fi

# 执行推理
python eval_single_img.py \
    --input_img "$INPUT_IMG" \
    --output_img "$OUTPUT_IMG" \
    --checkpoint "$CHECKPOINT" \
    --output_dxf "$OUTPUT_DXF" \
    --num_queries=2800 \
    --num_polys=70 \
    --semantic_classes=19
