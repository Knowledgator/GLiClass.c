
# ⭐GLiClass.c: Generalist and Lightweight Model for Sequence Classification in C

GLiClass.c is a C - based inference engine for running GLiClass(Generalist and Lightweight Model for Sequence Classification) models. This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.  

It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.
## 🛠 Build
```
git clone https://github.com/werent4/GLiClass.c.git
```

Then you need initialize and update submodules:
```
cd GLiClass.c
git submodule update --init --recursive
```
After that you need to download [ONNX runtime](https://github.com/microsoft/onnxruntime/releases) for your system.

Unpack it within the same derictory as GLiClass.c code.

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```
To use the GPU, you need to utilize the ONNX runtime with GPU support and set up cuDNN. Follow the instructions to install cuDNN here:  
https://developer.nvidia.com/cudnn-downloads 
```bash
tar -xvzf onnxruntime-linux-x64-gpu-1.19.2.tgz
```

Then create build directory and compile the project:
```bash
mkdir -p build
cd build
cmake ..
make -j8
```
Specific build for CPU
```bash
cmake -DCMAKE_BUILD_TYPE=Release -C ../CMakeLists_cpu.txt ..
make -j8
```

## 🚀 Run 
Before run make sure you have **jq** package installed 
```
sudo apt-get update
sudo apt-get install jq
```
This script will download all configurations for model from [Knowledgator Hugging Face](https://huggingface.co/knowledgator)
```
./run_GLiClass.sh knowledgator/gliclass-base-v1.0 /path/to/your_data.json
```
**Important** Data in your json file must be in folowing format
```
{
    "texts": [
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Hello"
    ],
    "labels": ["format","model","tool","cat"],
    "same_labels": true,
    "classification_type": "single-label" or "multi-label"
}
```
or
```
{
    "texts": [
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Hello"
    ],
    "labels": [
        ["format","model","tool","cat"],
        ["format","model","tool","cat"],
        ["format","model","tool","cat"],
        ]
    "same_labels": false,
    "classification_type": "single-label" or "multi-label"
}
```
## Convert your model
Run converting
```
python ONNX/convert_to_onnx.py \
        --model_path "knowledgator/gliclass-base-v1.0" \
        --save_path "model/" \
        --quantize True \
        --classification_type "multi-label"
```

Run test
```
python ONNX/test_onnx.py \
        --onnx_path "model/" \
        --test_quantized False
```