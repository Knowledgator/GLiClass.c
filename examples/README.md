# Examples

You can download the model from [HuggingFace](https://huggingface.co/models?search=gliclass) using [download_GLiClass_model.sh](download_GLiClass_model.sh) or [download_GliClass_model.ps1](download_GliClass_model.ps1) scripts:

```bash
./download_GLiClass_model.sh knowledgator/gliclass-base-v1.0
```

Examples' dependencies and build instructions are equivalent to those provided in [GLiClass.c/README.md](../README.md#-build).

To build all examples:

* Linux build
```bash
cmake -DONNX=ON -DONNX_CUDA=ON -DONNX_OPENVINO=ON -DOPENVINO=ON -DONNXRUNTIME_PATH="./onnxruntime-linux-x64-1.19.2" -DOPENVINO_PATH="/opt/intel/openvino/runtime" -S . -B build
cmake --build build -j
```

* Windows build:

```bash
cmake -DONNX=ON -DONNX_CUDA=ON -DONNX_OPENVINO=ON -DOPENVINO=ON  -DONNXRUNTIME_PATH="./onnxruntime-win-x64-1.19.2" -DOPENVINO_PATH="C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime" -S . -B build
cmake --build build -j --config Release
```

**Notice:** If you did not make ONNX accesible system-wide, you need to specify where the artifact is located using the ONNXRUNTIME_PATH property (full path or relative to the GLiClass.c directory). If you did not make OpenVino runtime accesible system-wide, you need to specify where the artifact is located using the OPENVINO_PATH property (full path or relative to the GLiClass.c directory)