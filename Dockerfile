FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    jq \
    libstdc++6 \
    libcjson-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY build/ ./build/
COPY tokenizer/ ./tokenizer/
COPY onnx/ ./onnx/
COPY onnxruntime-linux-x64-1.19.2/ ./onnxruntime-linux-x64-1.19.2/
RUN chmod +x ./build/GLiClass

# Last run parametr accepts only true or false values. Correct values may be found in onnx folder for specific GLiClass model
# Example: https://huggingface.co/knowledgator/gliclass-large-v1.0-init/blob/17791db81562aa56c72902b79341604522d67c59/onnx/config.json#L4
RUN echo '#!/bin/bash\n./build/GLiClass "$1" false' > /app/run_wrapper.sh && \
    chmod +x /app/run_wrapper.sh ./build/GLiClass

ENV LD_LIBRARY_PATH="/app/onnxruntime-linux-x64-1.19.2/lib"

RUN ldconfig

ENTRYPOINT ["/app/run_wrapper.sh"]
