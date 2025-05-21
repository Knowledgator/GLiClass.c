if [ -z "$1" ]; then
    echo "Usage: $0 <model_name> [trt_save_path] [precision] [batch_size]"
    echo "Example: $0 knowledgator/gliclass-base-v1.0 ./tensorrt FP16 1"
    exit 1
fi

MODEL_NAME="$1"
OUTPUT_DIR="${2:-./tensorrt}"
PRECISION="${3:-FP16}"  # Default FP16
BATCH_SIZE="${4:-1}"    # Default batch size 1

# Search for trtexec executable
TRTEXEC_PATHS=(
    "trtexec"                      
    "/usr/src/tensorrt/bin/trtexec" 
    "/usr/local/bin/trtexec"
    "/opt/tensorrt/bin/trtexec"
)

TRTEXEC=""
for path in "${TRTEXEC_PATHS[@]}"; do
    if command -v "$path" &> /dev/null || [ -x "$path" ]; then
        TRTEXEC="$path"
        echo "Found trtexec: $TRTEXEC"
        break
    fi
done

if [ -z "$TRTEXEC" ]; then
    echo "Error: trtexec not found. Checked paths: ${TRTEXEC_PATHS[*]}"
    echo "Specify full path to trtexec manually as an argument or modify TRTEXEC_PATHS in the script."
    exit 1
fi

# Create directory for TensorRT models
mkdir -p "$OUTPUT_DIR"

# Download model using existing script
echo "Downloading model $MODEL_NAME..."
./download_GLiClass_model.sh "$MODEL_NAME"

# Check that the model was downloaded successfully
ONNX_MODEL_PATH="./onnx/model.onnx"
if [ ! -f "$ONNX_MODEL_PATH" ]; then
    echo "Error: ONNX model not found at path $ONNX_MODEL_PATH"
    exit 1
fi

EINSUM_SCRIPT="./einsum_to_matmul.py"
if [ ! -f "$EINSUM_SCRIPT" ]; then
    echo "Error: Einsum conversion script not found at path $EINSUM_SCRIPT"
    exit 1
fi

# Create modified ONNX model with Einsum replaced by MatMul
FIXED_ONNX_MODEL_PATH="./onnx/model_fixed_einsum.onnx"
echo "Converting Einsum operation to MatMul using $EINSUM_SCRIPT..."
python3 "$EINSUM_SCRIPT" "$ONNX_MODEL_PATH" "$FIXED_ONNX_MODEL_PATH"

if [ ! -f "$FIXED_ONNX_MODEL_PATH" ]; then
    echo "Error: Failed to convert model with Einsum to MatMul replacement"
    echo "Continuing conversion with original model..."
    FIXED_ONNX_MODEL_PATH="$ONNX_MODEL_PATH"
else
    echo "Model conversion completed successfully!"
fi

# Configure trtexec launch parameters
TRT_ENGINE_PATH="$OUTPUT_DIR/$(basename "$MODEL_NAME" | tr '/' '_').engine"
PRECISION_FLAG=""
case $PRECISION in
    FP16)
        PRECISION_FLAG="--fp16"
        ;;
    INT8)
        PRECISION_FLAG="--int8"
        ;;
    *)
        # For FP32 or other non-standard formats
        PRECISION_FLAG=""
        ;;
esac

echo "Converting ONNX model to TensorRT ($PRECISION)..."
"$TRTEXEC" \
    --onnx="$FIXED_ONNX_MODEL_PATH" \
    --saveEngine="$TRT_ENGINE_PATH" \
    --minShapes=input_ids:${BATCH_SIZE}x1,attention_mask:${BATCH_SIZE}x1 \
    --optShapes=input_ids:${BATCH_SIZE}x1024,attention_mask:${BATCH_SIZE}x1024 \
    --maxShapes=input_ids:${BATCH_SIZE}x2048,attention_mask:${BATCH_SIZE}x2048 \
    --memPoolSize=workspace:1073741824 \
    $PRECISION_FLAG \
    --builderOptimizationLevel=0 \
    --maxTactics=1 \
    --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS \
    --versionCompatible \
    --excludeLeanRuntime \
    --useSpinWait

# Check that the conversion completed successfully
if [ $? -eq 0 ] && [ -f "$TRT_ENGINE_PATH" ]; then
    echo "Conversion completed successfully!"
    echo "TensorRT model saved: $TRT_ENGINE_PATH"
    
    # Copy configuration files
    cp "./onnx/config.json" "$OUTPUT_DIR/"
    cp -r "./tokenizer" "$OUTPUT_DIR/"
    
    echo "Configuration and tokenizer copied to $OUTPUT_DIR"
else
    echo "Error converting model to TensorRT."
    exit 1
fi

echo ""
echo "Done! TensorRT model generated with the following parameters:"
echo "- Batch size: $BATCH_SIZE"
echo "- Precision: $PRECISION"
echo "- Min input shapes: ${BATCH_SIZE}x1"
echo "- Opt input shapes: ${BATCH_SIZE}x1024"
echo "- Max input shapes: ${BATCH_SIZE}x2048"
echo "- Optimizations used: builderOptimizationLevel=0, maxTactics=1"
echo "- Memory: 1073741824 bytes (~1GB)"
echo ""
echo "Model path: $TRT_ENGINE_PATH"