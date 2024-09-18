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

Build
```
git clone url will be added after realese
```

Then you need initialize and update submodules:
```
cd GLiClass.c
git submodule update --init --recursive
```
Before run make sure you have jq package installed 
```
sudo apt-get update
sudo apt-get install jq
```

Run 
```
./run_GLiClass.sh  knowledgator/gliclass-small-v1.0
```