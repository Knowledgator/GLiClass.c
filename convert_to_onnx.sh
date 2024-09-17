#!/bin/bash

model_path="knowledgator/gliclass-large-v1.0-init"
save_path="onnx/"
quantize="True"
classification_type="multi-label"

# Convert
python ONNX_CONVERTING/convert_to_onnx.py \
    --model_path "$model_path" \
    --save_path "$save_path" \
    --quantize "$quantize" \
    --classification_type "$classification_type"
