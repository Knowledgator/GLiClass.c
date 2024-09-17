@echo off

set model_path=knowledgator/gliclass-small-v1.0-init
set save_path=onnx/
set quantize=True
set classification_type=multi-label
set test_quantized=False

:: Convert
python ONNX_CONVERTING/convert_to_onnx.py ^
    --model_path %model_path% ^
    --save_path %save_path% ^
    --quantize %quantize% ^
    --classification_type %classification_type%
