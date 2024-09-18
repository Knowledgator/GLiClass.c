#!/bin/bash
if [ -z "$1" ]; then
    echo "You need to specify model name e.g: ./run_GLiClass.sh knowledgator/gliclass-base-v1.0"
    exit 1
fi

MODEL_NAME="$1"

# Dir and tokenizer file
DIR="tokenizer"
FILE="$DIR/tokenizer.json"
TOKENIZER_URL="https://huggingface.co/$MODEL_NAME/raw/main/tokenizer.json"

# Check if we have file
if [ ! -f "$FILE" ]; then
    echo "File $FILE was not found. Downloading file..."
    mkdir -p "$DIR"
    wget -P "$DIR" "$TOKENIZER_URL"
else
    echo "File $FILE already exists. Updating ..."
    # rm $FILE
    # wget -P "$DIR" "$TOKENIZER_URL"
fi

# run
./build/GLiClass