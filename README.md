
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
    <a href="https://huggingface.co/collections/knowledgator/gliclass-6661838823756265f2ac3848">🤗 GliClass Collection</a>
</p>

## 🛠 Build
We currently support this providers:
 - ONNX Runtime;
 - OpenVino;

You can manage what support to include by this CMake options (can be combined):
 - ONNX: Build ONNX runtime inference with default CPU provider (enabled by default);
 - ONNX_CUDA: Build ONNX runtime inference with CUDA provider support;
 - ONNX_OPENVINO: Build ONNX runtime inference with OpenVino provider support;
 - OPENVINO: Build Openvino runtime inference;

**NOTICE:** For ONNX runtime you can also use other providers. See example /examples/inference_custom_ort.c for how to use providers that do not have prebuild init.

Other build options:
 - ONNXRUNTIME_PATH: Required for ONNX provider builds if ONNX runtime is not in path;
 - OPENVINO_PATH: Required for OpenVino provider builds if OpenVino runtime is not in path;
 - OPENVINO_LIB_PATH: Overwrite default OpenVino lib dir that resolved from provided path. Ignored if OPENVINO_PATH wasn't provided. Default values:
    - for Linux: ${OPENVINO_PATH}/lib/;
    - for Windows: ${OPENVINO_PATH}/lib/intel64/Release;
 - SKIP_CHECK: Skip environment check for dependencies;

### **📦 Common build dependencies**
 - CMake (>= 3.19)
 - [Rust](https://www.rust-lang.org/tools/install)
 - OpenMP

### **📦 ONNX runtime dependencies & instructions**
**Dependencies:**
 - [Common build dependencies](#-common-build-dependencies)
 - [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) CPU version for your system

To build the project for CPU, you can use the standard version of ONNXRuntime without GPU support (assets archives do not have 'gpu' suffix in name, see examples below). Make sure you download and unzip ONNX runtime archive or build it from source.

Clone repo with:

```
git clone https://github.com/Knowledgator/GLiClass.c.git
```

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```
Then compile the project. If you did not make ONNX accesible system-wide, you need to specify where the artifact is located using the ONNXRUNTIME_PATH property (full path or relative to the GLiClass.c directory):

* Linux build
```bash
cmake -DONNX=ON -DONNXRUNTIME_PATH="./onnxruntime-linux-x64-1.19.2" -S . -B build
cmake --build build -j
```

* Windows build:

```bash
cmake -DONNX=ON -DONNXRUNTIME_PATH="./onnxruntime-win-x64-1.19.2" -S . -B build
cmake --build build -j --config Release
```

**NOTICE:** Some issues may occur related to overriding the ONNX Runtime library. To fix this, you can copy the .dll files from the ONNX Runtime lib\ directory to the build directory.

**NOTICE:** This build and other GLiClass ONNX builds also can be used with other ONNX providers. For more details how to use it with providers that do not have prebuild init see [examples/inference_custom_ort.c](examples/inference_custom_ort.c).

### **📦 ONNX runtime CUDA GPU build dependencies & instruction**
**Dependencies:**
 - [Common build dependencies](#-common-build-dependencies)
 - [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases) GPU version for yor system
 - [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
 - [cuDNN](https://developer.nvidia.com/cudnn-downloads)

To build the project for GPU, you need to install NVIDIA and cuDNN drivers. Make sure you download and unzip ONNX runtime archive or build it from source.  

Clone repo with:

```
git clone https://github.com/Knowledgator/GLiClass.c.git
```

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-gpu-1.19.2.tgz 
```

Then compile the project. If you did not make ONNX accesible system-wide, you need to specify where the artifact is located using the ONNXRUNTIME_PATH property (full path or relative to the GLiClass.c directory):

* Linux build
```bash
cmake -DONNX_CUDA=ON -DONNXRUNTIME_PATH="./onnxruntime-linux-x64-gpu-1.19.2" -S . -B build
cmake --build build -j
```

* Windows build:

```bash
cmake -DONNX_CUDA=ON -DONNXRUNTIME_PATH="./onnxruntime-win-gpu-x64-1.19.2" -S . -B build
cmake --build build -j --config Release
```

**NOTICE:** Some issues may occur related to overriding the ONNX Runtime library. To fix this, you can copy the .dll files from the ONNX Runtime lib\ directory to the build directory.

**NOTICE:** This build and other GLiClass ONNX builds also can be used with other ONNX providers. For more details how to use it with providers that do not have prebuild init see [examples/inference_custom_ort.c](examples/inference_custom_ort.c).

### **📦 ONNX runtime OpenVino provider build dependencies & instruction**
**Dependencies:**
 - [Common build dependencies](#-common-build-dependencies)
 - [OpenVino](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT)
 - [ONNXRuntime](https://github.com/microsoft/onnxruntime/releases)
 - [ONNXRuntime OpenVino provider](https://github.com/intel/onnxruntime/releases)

To build the project with ONNX OpenVino provider, you need to install OpenVino runtime. This requires [ONNXRuntime OpenVino provider](https://github.com/intel/onnxruntime).

* For Windows you can use full ONNX build from this fork or just download onnxruntime_providers_openvino.dll;
* Other system will require to build it from source;

**NOTICE:** Ensure compatibility between ONNX Runtime and onnxruntime_providers_openvino.dll (for Windows users). Info about compatibility can be found in [release notes to fork](https://github.com/intel/onnxruntime/releases).

Clone repo with:

```
git clone https://github.com/Knowledgator/GLiClass.c.git
```

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```

Then compile the project. If you did not make ONNX accesible system-wide, you need to specify where the artifact is located using the ONNXRUNTIME_PATH property (full path or relative to the GLiClass.c directory). If you did not make OpenVino runtime accesible system-wide, you need to specify where the artifact is located using the OPENVINO_PATH property (full path or relative to the GLiClass.c directory):

* Linux build
```bash
cmake -DONNX_OPENVINO=ON -DONNXRUNTIME_PATH="./onnxruntime-linux-x64-1.19.2" -DOPENVINO_PATH="/opt/intel/openvino/runtime" -S . -B build
cmake --build build -j
```

* Windows build:

```bash
cmake -DONNX_OPENVINO=ON -DONNXRUNTIME_PATH="./onnxruntime-win-x64-1.19.2" -DOPENVINO_PATH="C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime" -S . -B build
cmake --build build -j --config Release
```

**NOTICE:** Some issues may occur related to overriding the ONNX Runtime library. To fix this, you can copy the .dll files from the ONNX Runtime lib\ directory to the build directory.

**NOTICE:** This build and other GLiClass ONNX builds also can be used with other ONNX providers. For more details how to use it with providers that do not have prebuild init see [examples/inference_custom_ort.c](examples/inference_custom_ort.c).

### **📦 OpenVino runtime build dependencies & instruction**
**Dependencies:**
 - [Common build dependencies](#-common-build-dependencies)
 - [OpenVino](https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT)

Clone repo with:

```
git clone https://github.com/Knowledgator/GLiClass.c.git
```

Compile the project. If you did not make OpenVino runtime accesible system-wide, you need to specify where the artifact is located using the OPENVINO_PATH property (full path or relative to the GLiClass.c directory):

* Linux build
```bash
cmake -DONNX=OFF -DOPENVINO=ON -DOPENVINO_PATH="/opt/intel/openvino/runtime" -S . -B build
cmake --build build -j
```

* Windows build:

```bash
cmake -DONNX=OFF -DOPENVINO=ON -DOPENVINO_PATH="C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime" -S . -B build
cmake --build build -j --config Release
```

**NOTICE:** This build and other GLiClass ONNX builds also can be used with other ONNX providers. For more details how to use it with providers that do not have prebuild init see [examples/inference_custom_ort.c](examples/inference_custom_ort.c).

## 🚀 Quick start

### Inference configuration
```c
GLiClassInferenceConfig config;
GLiClassStatus* status = gliclass_create_inference_config(
    2, // batch size
    0, // min length (tokens) wich can be processed
    2048, // max length (tokens) wich can be processed
    0.5, // score threshold
    "multi-label", // classification type. Can be one of: "multi-label", "single-label"
    true, // add_space_prefix parameter
    &config // output
);
if (status != NULL) {
    fprintf(stderr, "Invalid config: %s", status->msg);
    gliclass_free_status(status);
    return 1;
}
```

### Session initialization (prebuild)
```c
int num_threads= 8;
GLiClassSession* session = NULL;
status = gliclass_init(
    model_path, // path to model. Expected *.onnx (ONNX model) OR *.xml extension (Compiled OpenVino model. *.bin file should also exist with same name at the path)
    model_config_path, // model config path 
    tokenizer_path, // tokenizer path
    num_threads, // max number of threads that can be used
    false, // use mutex on provider iference. Can be usefull with multithreading
    GC_ONNX_OPENVINO, // Provider type
    GC_CPU, // Device type
    &session // output
);
if (status != NULL) {
    fprintf(stderr, "Unable to create session: %s", status->msg);
    gliclass_free_status(status);
    return 1;
}
```

### Session initialization (custom provider)
See [examples/inference_custom_ort.c](examples/inference_custom_ort.c) for more details.

### Inference (single)
```c
// Inputs
const char* text = "ONNX is an open-source format designed to enable the interoperability of AI models.";
const char* labels[] = {"format","model","tool","necessity"};
const size_t num_labels = 4;

// Outputs
GLiClassResult* results = NULL;
GLiClassTokensInfo info;
size_t num_results = 0;

status = gliclass_infer(
    session,
    &config,
    text,
    labels,
    num_labels,
    &results,
    &num_results,
    &info // NULL can be provided if you do not need info about tokenization
);
if (status != NULL) {
    fprintf(stderr, "Error during inference: %s", status->msg);
    gliclass_free_status(status);
    gliclass_cleanup(session);
    return 1;
}

fprintf(stdout, "\nText: %s\n", text);
fprintf(stdout, "\nTruncated: %s\n", info.truncated ? "true": "false");
fprintf(stdout, "\nProcessed tokens: %zu\n", info.tokens_num);
for (size_t i = 0; i < num_results; i++) {
    fprintf(stdout, "Label_%zu: %s, score: %f\n", i, results[i].label, results[i].score);
}
gliclass_free_results(results, num_results);
gliclass_cleanup(session);
return 0;
```

### Inference (batch)
```c
// Inputs
const char* texts[] = {
    "ONNX is an open-source format designed to enable the interoperability of AI models.",
    "Why are you running?",
    "Support Ukraine"
};
size_t num_texts = 3;

const char* group1[] = {"format","model","tool","cat"};
const char* group2[] = {"question","tool","statement"};
const char* group3[] = {"call to action", "necessity"};
const char** labels[] = {group1, group2, group3};

const size_t labels_shape[] = {4, 3, 2};
const size_t labels_shape_size = 3;

// Outputs
GLiClassResult** results = NULL;
size_t* results_shape = NULL;
size_t results_shape_size = 0;

status = gliclass_infer_batch(
    session, 
    &config,
    texts, 
    num_texts, 
    labels, 
    labels_shape, 
    labels_shape_size,
    &results,
    &results_shape,
    &results_shape_size,
    NULL
);
if (status != NULL) {
    fprintf(stderr, "Error during inference: %s", status->msg);
    gliclass_free_status(status);
    gliclass_cleanup(session);
    return 1;
}

for (size_t i = 0; i < results_shape_size; i++) {
    fprintf(stdout, "\nText_%zu/%zu: %s\n", i, results_shape_size, texts[i]);
    for (size_t j = 0; j < results_shape[i]; j++) {
        fprintf(stdout, "Label_%zu: %s, score: %f\n", j, results[i][j].label, results[i][j].score);
    }      
}

gliclass_free_results_batch(results, results_shape, results_shape_size);
gliclass_cleanup(session);
return 0;
```

## Examples

More examples provided in [examples](examples/) directory. Follow instruction in [README.md](examples/README.md) to compile and run examples.

## Docker 
Also, some GLiClass models already have their own dockerized version, you can find them on our [official dockerhub](https://hub.docker.com/repositories/knowledgator)
  
The general principle of using dockerized models
1. Pull the image:
    ```bash
    docker pull knowledgator/gliclass-specific-version
    ```
2. Run the container:
    ```bash
    docker run -v /path/to/folder_with_data:/app/data knowledgator/gliclass-specific-version /app/your_data.json
    ```  
More detailed instructions are available in the Docker Hub repositories.

## Convert your model
If the GLiClass model you need does not yet have an ONNX version, you can create it yourself using our script.
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
