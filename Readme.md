
# ⭐GLiClass.c: Generalist and Lightweight Model for Sequence Classification in C

GLiClass.c is a C - based inference engine for running GLiClass(Generalist and Lightweight Model for Sequence Classification) models. This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.  

It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.

<p align="center">
    <img src="kg.png" style="position: relative; top: 5px;">
    <a href="https://www.knowledgator.com/"> Knowledgator</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.linkedin.com/company/knowledgator/">✔️ LinkedIn</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/NNwdHEKX">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/knowledgator/GLiClass_SandBox">🤗 Space</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?library=gliner&sort=trending">🤗 GliClass Collection</a>
</p>

## 🛠 Build
We have provided 2 types of build for CPU and GPU, below are the requirements and steps necessary for successful build.

```
git clone https://github.com/werent4/GLiClass.c.git
```

Then you need initialize and update submodules:
```
cd GLiClass.c
git submodule update --init --recursive
```

### <img src="https://github.com/user-attachments/assets/4d2fd37f-9882-4fea-902b-be5ccc1edae2" alt="image" height="30" width="30"> CPU build dependencies & instructions
 - CMake (>= 3.25)
 - [Rust](https://www.rust-lang.org/tools/install)
 - [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) CPU version for yor system
 - OpenMP 

To build the project for CPU, use the standard version of ONNXRuntime without GPU support. Make sure you download and unzip ```onnxruntime-linux-x64-1.19.2``` into the same directory as the GliClass code.  

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```
Then create build directory and compile the project:  
```bash
mkdir -p build
cd build
cmake  -DBUILD_TARGET=CPU ..
make -j8
```
### <img src="https://github.com/user-attachments/assets/92a49538-feb0-4fcb-8789-8d6edfc2ceed" alt="image" height="40" width="40"> GPU build dependencies & instruction
 - CMake (>= 3.25)
 - [Rust](https://www.rust-lang.org/tools/install)
 - [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) GPU version for yor system
 - OpenMP
 - NVIDIA GPU
 - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
 - [cuDNN](https://developer.nvidia.com/cudnn-downloads )

To build the project for GPU, you need to install NVIDIA and cuDNN drivers. Make sure you download and unzip ```onnxruntime-linux-x64-gpu-1.19.2``` into the same directory as the GliClass code.  

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-gpu-1.19.2.tgz 
```
Then create build directory and compile the project:  
```bash
mkdir -p build
cd build
cmake  -DBUILD_TARGET=GPU ..
make -j8
```

## 🚀 Run 
There are 2 options for launching:
 - ```run_GLiClass.sh``` (automatically configures many dependencies)
 - Manual setup 

### run_GLiClass.sh
Running via run_GLiClass.sh requires the additional **jq** module  
```
sudo apt-get update
sudo apt-get install jq
```
This script will download all configurations for model from [Knowledgator GLiClass collection](https://huggingface.co/collections/knowledgator/gliclass-6661838823756265f2ac3848). You only need to specify the model name and the path to the data that needs to be classified e.g.
```
./run_GLiClass.sh knowledgator/gliclass-base-v1.0 /path/to/your_data.json
```
**Note** some models can not be loaded with this script, manual configuration is required to run them.  
The list of such models is given below  
 - knowledgator/gliclass-qwen-1.5B-v1.0
 - knowledgator/gliclass-llama-1.3B-v1.0

### Manual setup
To start manual configuration you need to download the ONNX version of the model from [Knowledgator GLiClass collection](https://huggingface.co/collections/knowledgator/gliclass-6661838823756265f2ac3848) and place it in a directory convenient for you. By default, the program searches in the ```onnx``` directory, but the directory can be changed in the ```include/paths.h``` file.  
Next you need to download the tokenizer configuration file ```tokenizer.json```. By default, the program searches in the ```tokenizer``` directory, but the directory can be changed in the ```include/paths.h``` file as well. 

``` C
// include/paths.h
#define TOKENIZER_PATH "tokenizer/tokenizer.json" // Path to tokenizer file (JSON configuration)
#define MODEL_PATH "onnx/model.onnx"              // Path to ONNX model for inference
```

Parameters such as **batch size**, **max length**, **decision threshold** and **number of threads** (for CPU build) can be configured in the ```include/configs.h``` file.
``` C
// include/configs.h
#define BATCH_SIZE 8    // Number of texts in one batch for processing by the model
#define MAX_LENGTH 1024 // Maximum length of tokenized text (number of tokens)
#define THRESHOLD 0.5f  // Threshold for making a classification decision 
#define NUM_THREADS 8   // Number of threads for CPU (does not affect GPU performance)
```

After all the necessary configurations, the program can be launched with the following command  
``` bash
./build/GLiClass /path/to/your_data.json [prompt_first: true/false]
```
**Note** the value for ```prompt_first``` parameter can be found in the ```config.json``` [configuration file for the onnx version](https://huggingface.co/knowledgator/gliclass-small-v1.0/blob/be5ffb291f2fa96fed865390ceee092efebf4b13/onnx/config.json#L4).

**Important** Data in your json file must be in folowing format
```json
{
    "texts": [
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Support Ukraine"
    ],
    "labels": ["format","model","tool","necessity"],
    "same_labels": true,
    "classification_type": "multi-label"
}
```
or
```json
{
    "texts": [
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Support Ukraine"
    ],
    "labels": [
        ["format","model","tool","cat"],
        ["question","tool","statement"],
        ["call to action", "necessity"],
        ]
    "same_labels": false,
    "classification_type": "single-label" 
}
```
## Convert your model
If the GLiClass model you need does not yet have an onnx version, you can create it yourself using our script.
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
