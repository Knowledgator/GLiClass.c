```
python .\ONNX\convert_to_onnx.py --model_path "knowledgator/gliclass-base-v1.0" --save_path "model/" --quantize True --classification_type "multi-label"

python .\ONNX\test_onnx.py
```