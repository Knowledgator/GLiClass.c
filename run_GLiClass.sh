#!/bin/bash
if [ -z "$1" ]; then
    echo "You need to specify model name e.g: ./run_GLiClass.sh knowledgator/gliclass-base-v1.0"
    exit 1
fi

MODEL_NAME="$1"

# Dirs
TOKENIZER_DIR="tokenizer"
MODEL_DIR="onnx"

mkdir -p "$MODEL_DIR"
mkdir -p "$TOKENIZER_DIR"

# Files
TOKENIZER_FILE="$TOKENIZER_DIR/tokenizer.json"
MODEL_CONFIG_FILE="$MODEL_DIR/config.json"
MODEL_ONNX_FILE="$MODEL_DIR/model.onnx"

# URLS
MODEL_CONFIG_URL="https://huggingface.co/$MODEL_NAME/resolve/main/onnx/config.json"
MODEL_ONNX_URL="https://huggingface.co/$MODEL_NAME/resolve/main/onnx/model.onnx"
TOKENIZER_URL="https://huggingface.co/$MODEL_NAME/raw/main/tokenizer.json"

download_file() {
    local file_type=$1
    local directory=$2
    local url=$3

    echo "Downloading $file_type..."
    wget -P "$directory" "$url"
}

download_model() {
    rm -rf $TOKENIZER_FILE
    rm -rf $MODEL_CONFIG_FILE
    rm -rf $MODEL_ONNX_FILE

    download_file "configuration for the model" "$MODEL_DIR" "$MODEL_CONFIG_URL"
    download_file "ONNX version of model" "$MODEL_DIR" "$MODEL_ONNX_URL"
    download_file "tokenizer config" "$TOKENIZER_DIR" "$TOKENIZER_URL"    
}

# Download logic
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "File $MODEL_CONFIG_FILE was not found."
    download_model
else
    MODEL_TYPE=$(jq -r '.original_model_name' "$MODEL_CONFIG_FILE")
    if [ "$MODEL_TYPE" != "$MODEL_NAME" ]; then
        echo "Reconfigurating the model"
        download_model
    else
        echo "Checking the integrity of model files"
        FILES=($TOKENIZER_FILE $MODEL_ONNX_FILE)
        for FILE in "${FILES[@]}"; do
            if [ ! -f "$FILE" ]; then
                echo "Missing file: $FILE. Downloading..."

                if [ "$FILE" == "$TOKENIZER_FILE" ]; then
                    download_file "tokenizer config" "$TOKENIZER_DIR" "$TOKENIZER_URL"
                elif [ "$FILE" == "$MODEL_ONNX_FILE" ]; then
                    download_file "ONNX version of model" "$MODEL_DIR" "$MODEL_ONNX_URL"
                fi
            fi   
        done        

    fi
    echo "Everithing was set up. Running inference"
fi

# run
./build/GLiClass