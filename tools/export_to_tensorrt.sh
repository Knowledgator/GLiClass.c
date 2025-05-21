# #!/bin/bash

# # Проверка аргументов
# if [ -z "$1" ]; then
#     echo "Использование: $0 <имя_модели> [путь_сохранения_trt] [precision] [batch_size]"
#     echo "Пример: $0 knowledgator/gliclass-base-v1.0 ./tensorrt FP16 1"
#     exit 1
# fi

# # Присваиваем аргументы переменным
# MODEL_NAME="$1"
# OUTPUT_DIR="${2:-./tensorrt}"
# PRECISION="${3:-FP16}"  # По умолчанию FP16
# BATCH_SIZE="${4:-1}"    # По умолчанию размер батча 1

# # Поиск исполняемого файла trtexec
# TRTEXEC_PATHS=(
#     "trtexec"                      
#     "/usr/src/tensorrt/bin/trtexec" 
#     "/usr/local/bin/trtexec"
#     "/opt/tensorrt/bin/trtexec"
# )

# TRTEXEC=""
# for path in "${TRTEXEC_PATHS[@]}"; do
#     if command -v "$path" &> /dev/null || [ -x "$path" ]; then
#         TRTEXEC="$path"
#         echo "Найден trtexec: $TRTEXEC"
#         break
#     fi
# done

# if [ -z "$TRTEXEC" ]; then
#     echo "Ошибка: trtexec не найден. Проверены пути: ${TRTEXEC_PATHS[*]}"
#     echo "Укажите полный путь к trtexec вручную в аргументе или измените TRTEXEC_PATHS в скрипте."
#     exit 1
# fi

# # Создаем директорию для TensorRT моделей
# mkdir -p "$OUTPUT_DIR"

# # Скачиваем модель с помощью существующего скрипта
# echo "Скачиваем модель $MODEL_NAME..."
# ./download_GLiClass_model.sh "$MODEL_NAME"

# # Проверяем, что модель была успешно скачана
# ONNX_MODEL_PATH="./onnx/model.onnx"
# if [ ! -f "$ONNX_MODEL_PATH" ]; then
#     echo "Ошибка: Модель ONNX не найдена по пути $ONNX_MODEL_PATH"
#     exit 1
# fi

# EINSUM_SCRIPT="./einsum_to_matmul.py"
# if [ ! -f "$EINSUM_SCRIPT" ]; then
#     echo "Ошибка: Скрипт преобразования Einsum не найден по пути $EINSUM_SCRIPT"
#     exit 1
# fi

# # Создаем модифицированную ONNX модель с заменой Einsum на MatMul
# FIXED_ONNX_MODEL_PATH="./onnx/model_fixed_einsum.onnx"
# echo "Преобразуем Einsum операцию в MatMul используя $EINSUM_SCRIPT..."
# python3 "$EINSUM_SCRIPT" "$ONNX_MODEL_PATH" "$FIXED_ONNX_MODEL_PATH"

# if [ ! -f "$FIXED_ONNX_MODEL_PATH" ]; then
#     echo "Ошибка: Не удалось преобразовать модель с заменой Einsum на MatMul"
#     echo "Продолжаем конвертацию с оригинальной моделью..."
#     FIXED_ONNX_MODEL_PATH="$ONNX_MODEL_PATH"
# else
#     echo "Преобразование модели завершено успешно!"
# fi

# # Конфигурируем параметры запуска trtexec
# TRT_ENGINE_PATH="$OUTPUT_DIR/$(basename "$MODEL_NAME" | tr '/' '_').engine"
# PRECISION_FLAG=""
# case $PRECISION in
#     FP16)
#         PRECISION_FLAG="--fp16"
#         ;;
#     INT8)
#         PRECISION_FLAG="--int8"
#         ;;
#     *)
#         # Для FP32 или других нестандартных форматов
#         PRECISION_FLAG=""
#         ;;
# esac

# echo "Конвертируем ONNX модель в TensorRT ($PRECISION)..."
# "$TRTEXEC" \
#     --onnx="$FIXED_ONNX_MODEL_PATH" \
#     --saveEngine="$TRT_ENGINE_PATH" \
#     --minShapes=input_ids:${BATCH_SIZE}x1,attention_mask:${BATCH_SIZE}x1 \
#     --optShapes=input_ids:${BATCH_SIZE}x1024,attention_mask:${BATCH_SIZE}x1024 \
#     --maxShapes=input_ids:${BATCH_SIZE}x2048,attention_mask:${BATCH_SIZE}x2048 \
#     --memPoolSize=workspace:1073741824 \
#     $PRECISION_FLAG \
#     --builderOptimizationLevel=0 \
#     --maxTactics=1 \
#     --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS \
#     --versionCompatible \
#     --excludeLeanRuntime \
#     --useSpinWait

# # Проверяем, что конвертация завершилась успешно
# if [ $? -eq 0 ] && [ -f "$TRT_ENGINE_PATH" ]; then
#     echo "Конвертация успешно завершена!"
#     echo "Модель TensorRT сохранена: $TRT_ENGINE_PATH"
    
#     # Копируем конфигурационные файлы
#     cp "./onnx/config.json" "$OUTPUT_DIR/"
#     cp -r "./tokenizer" "$OUTPUT_DIR/"
    
#     echo "Конфигурация и токенизатор скопированы в $OUTPUT_DIR"
# else
#     echo "Ошибка при конвертации модели в TensorRT."
#     exit 1
# fi

# echo ""
# echo "Готово! Модель TensorRT сгенерирована со следующими параметрами:"
# echo "- Размер батча: $BATCH_SIZE"
# echo "- Точность: $PRECISION"
# echo "- Мин. размеры входа: ${BATCH_SIZE}x1"
# echo "- Опт. размеры входа: ${BATCH_SIZE}x1024"
# echo "- Макс. размеры входа: ${BATCH_SIZE}x2048"
# echo "- Использованы оптимизации: builderOptimizationLevel=0, maxTactics=1"
# echo "- Память: 1073741824 байт (~1GB)"
# echo ""
# echo "Путь к модели: $TRT_ENGINE_PATH"

#!/bin/bash

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name> [trt_save_path] [precision] [batch_size]"
    echo "Example: $0 knowledgator/gliclass-base-v1.0 ./tensorrt FP16 1"
    exit 1
fi

# Assign arguments to variables
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