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

RUN echo '#!/bin/bash\n./build/GLiClass "$1" false' > /app/run_wrapper.sh && \
    chmod +x /app/run_wrapper.sh ./build/GLiClass

ENV LD_LIBRARY_PATH="/app/onnxruntime-linux-x64-1.19.2/lib"

RUN ldconfig

ENTRYPOINT ["/app/run_wrapper.sh"]