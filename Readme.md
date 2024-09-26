
# ⭐GLiClass.c: Generalist and Lightweight Model for Sequence Classification in C

GLiClass.c is a C - based inference engine for running GLiClass(Generalist and Lightweight Model for Sequence Classification) models. This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.  

It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.

<p align="center">
    <img src="kg.png" style="position: relative; top: 5px;">
    <a href="https://www.knowledgator.com/"> Knowledgator</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" style="position: relative; top: 5px;" ><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
    <a href="https://www.linkedin.com/company/knowledgator/"> LinkedIn</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/NNwdHEKX">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/knowledgator/GLiClass_SandBox">🤗 Space</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?library=gliner&sort=trending">🤗 GliClass Collection</a>
</p>

## 🛠 Build
```
git clone https://github.com/werent4/GLiClass.c.git
```

Then you need initialize and update submodules:
```
cd GLiClass.c
git submodule update --init --recursive
```

<div style="display: flex; justify-content: space-between;">
    <div style="width: 48%;">
        <!-- CPU build dependencies -->
        <h3>CPU build dependencies</h3>
        <ul>
            <li>CMake (>= 3.18)</li>
            <li>ONNXRuntime CPU version</li>
            <li>OpenMP</li>
            <li>cJSON library</li>
            <li>tokenizers-cpp</li>
        </ul>
        <p>Инструкции<code>onnxruntime-linux-x64-1.19.2</code> в ту же директорию, что и проектный код.</p>
        <pre><code> cmake -DBUILD_TARGET=CPU ..</code></pre>
        <p>После этого запустите сборку:</p>
        <pre><code> make -j8 </code></pre>
    </div>
    <div style="width: 48%;">
        <!-- GPU build dependencies -->
        <h3>GPU build dependencies</h3>
        <ul>
            <li>CMake (>= 3.18)</li>
            <li>NVIDIA GPU + CUDA Toolkit</li>
            <li>cuDNN</li>
            <li>ONNXRuntime GPU version</li>
            <li>OpenMP</li>
            <li>cJSON library</li>
            <li>tokenizers-cpp</li>
        </ul>
        <p>Для сборки проекта для GPU вам необходимо установить драйверы NVIDIA и <code>cuDNN</code>. Загрузите и распакуйте <code>onnxruntime-linux-x64-gpu-1.19.2</code> в ту же директорию, что и проектный код.</p>
        <pre><code> cmake -DBUILD_TARGET=GPU ..</code></pre>
        <p>После этого запустите сборку:</p>
        <pre><code> make -j8 </code></pre>
    </div>
</div>

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
This script will download all configurations for model from [Knowledgator GLiClass collection](https://huggingface.co/collections/knowledgator/gliclass-6661838823756265f2ac3848)
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