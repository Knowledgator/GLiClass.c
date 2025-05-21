# Tools

This folder contains utilities for working with GLiClass models.

## Contents

- `download_GLiClass_model.sh` - Bash script for downloading GLiClass models from Hugging Face (Linux/macOS)
- `download_GliClass_model.ps1` - PowerShell script for downloading GLiClass models from Hugging Face (Windows)
- `einsum_to_matmul.py` - Python script for converting Einsum operations to MatMul in ONNX models
- `export_to_tensorrt.sh` - Script for exporting ONNX models to TensorRT format
- `to_trt.sh` - Example commands for converting models to TensorRT with various parameters

## Usage

### Downloading a model

Linux/macOS:
```bash
./download_GLiClass_model.sh knowledgator/gliclass-base-v1.0
```

Windows:
```powershell
.\download_GliClass_model.ps1 knowledgator/gliclass-base-v1.0
```
### Converting Einsum to MatMul

```bash
python3 einsum_to_matmul.py input_model.onnx output_model.onnx
```

### Exporting to TensorRT

```bash
./export_to_tensorrt.sh knowledgator/gliclass-base-v1.0 ./tensorrt FP16 1
```

## TODO

- Add export script for Windows