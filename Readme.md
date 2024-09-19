# Info
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

# Build
```
git clone url_will_be_added_after_realese
```

Then you need initialize and update submodules:
```
cd GLiClass.c
git submodule update --init --recursive
mkdir -p build
cd build
cmake ..
make
```
Before run make sure you have jq package installed 
```
sudo apt-get update
sudo apt-get install jq
```

# Run 
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
    "same_labels": true
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
    "same_labels": true
}
```
