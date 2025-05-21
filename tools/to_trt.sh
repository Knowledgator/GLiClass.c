# /usr/src/tensorrt/bin/trtexec \ # 32GB
#     --onnx=./examples/onnx/model_fixed_einsum.onnx \
#     --saveEngine=test.engine \
#     --shapes=input_ids:1x128,attention_mask:1x128 \
#     --memPoolSize=workspace:4294967296 \
#     --fp16 \
#     --builderOptimizationLevel=2 \
#     --tacticSources=-CUDNN,+CUBLAS_LT \
#     --tempfileControls=in_memory:allow,temporary:deny \
#     --profilingVerbosity=none \
#     --verbose
#     --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,+JIT_CONVOLUTIONS \
#     --maxTactics=5 

# /usr/src/tensorrt/bin/trtexec \ # 32GB
#     --onnx=./examples/onnx/model_fixed_einsum.onnx \
#     --saveEngine=test.engine \
#     --shapes=input_ids:1x128,attention_mask:1x128 \
#     --memPoolSize=workspace:4294967296 \
#     --fp16 \
#     --builderOptimizationLevel=2 \
#     --profilingVerbosity=none \
#     --tacticSources=-CUDNN,+CUBLAS_LT \
#     --tempfileControls=in_memory:allow,temporary:deny \
#     --verbose

###############################################################
/usr/src/tensorrt/bin/trtexec \ 
    --onnx=examples/onnx/model_fixed_einsum.onnx \
    --saveEngine=test_32_vc.engine  \
    --shapes=input_ids:1x32,attention_mask:1x32  \
    --memPoolSize=workspace:1073741824  \
    --fp16  \
    --builderOptimizationLevel=0  \
    --maxTactics=1  \
    --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS  \
    --versionCompatible  \
    --excludeLeanRuntime \
    --useSpinWait \

/usr/src/tensorrt/bin/trtexec --onnx=examples/onnx/model_fixed_einsum.onnx --saveEngine=test_128_vc.engine --shapes=input_ids:1x128,attention_mask:1x128  --memPoolSize=workspace:1073741824 --fp16 --builderOptimizationLevel=0 --maxTactics=1 --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS --versionCompatible --excludeLeanRuntime --useSpinWait 

# dynamic shapes
/usr/src/tensorrt/bin/trtexec \
    --onnx=examples/onnx/model_fixed_einsum.onnx \
    --saveEngine=test_dynamic_vc.engine \
    --minShapes=input_ids:1x1,attention_mask:1x1 \
    --optShapes=input_ids:1x1024,attention_mask:1x1024 \
    --maxShapes=input_ids:1x2048,attention_mask:1x2048 \
    --memPoolSize=workspace:1073741824 \
    --fp16 \
    --builderOptimizationLevel=0 \
    -maxTactics=1 \
    --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS \
    --versionCompatible \
    --excludeLeanRuntime \
    --useSpinWait

# /usr/src/tensorrt/bin/trtexec \
#     --onnx=./examples/onnx/model_fixed_einsum.onnx \
#     --saveEngine=test_64.engine \
#     --shapes=input_ids:1x64,attention_mask:1x64 \
#     --memPoolSize=workspace:1073741824 \
#     --fp16 \
#     --builderOptimizationLevel=0 \
#     --maxTactics=1 \
#     --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS