import onnxruntime, argparse, json
import torch, os
import numpy as np
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from typing import Tuple, List

def load_config(load_path: str) -> Tuple[str, str, List[float]]:
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_model_name = data["original_model_name"]
    architecture_type = data["architecture_type"]
    original_logits = data["original_logits"]
    
    return original_model_name, architecture_type, original_logits

def prepare_onnx_inputs(pipeline, text, labels):
    inputs = pipeline.pipe.prepare_inputs(text, labels)
    return {
        ort_session.get_inputs()[0].name: np.array(inputs["input_ids"]),
        ort_session.get_inputs()[1].name: np.array(inputs["attention_mask"])
    }

def run_inference_and_compare(ort_session, onnx_inputs, original_logits) -> bool:
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_tensor = torch.tensor(onnx_outputs[0])
    print(f"ONNX Outputs: {onnx_tensor}")

    comparison_result = torch.allclose(original_logits, onnx_tensor, atol=1e-4)
    return comparison_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default = 'model/')
    parser.add_argument('--test_quantized', type=bool, default = False)

    args = parser.parse_args()

    files = os.listdir(args.onnx_path)
    if "config.json" not in files:
        raise FileNotFoundError("Configuration file is missing")
    
    original_model_name, architecture_type, original_logits = load_config(args.onnx_path + "config.json")
    original_logits = torch.Tensor(original_logits)

    if args.test_quantized:
        onnx_model = next((file for file in files if "quantized.onnx" in file), None)
    else:
        onnx_model = next((file for file in files if "model.onnx" in file), None)


    # Step 1: Loading the ONNX Model
    onnx_model_path = args.onnx_path + onnx_model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Step 2: Tokenize text
    gliclass_model = GLiClassModel.from_pretrained(original_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    pipeline = ZeroShotClassificationPipeline(gliclass_model, tokenizer, classification_type="multi-label", device="cpu")
    print("Model loaded")

    text = "ONNX is an open-source format designed to enable the interoperability of AI models across various frameworks and tools."
    labels = ['format', 'model', 'tool', 'cat']

    # Step 3: Preparing Input Data for ONNX Model and testing
    onnx_inputs = prepare_onnx_inputs(pipeline, text, labels)
    print("Data prepared")

    # Step 4: Run Inference and compare logits
    comparison_result = run_inference_and_compare(ort_session, onnx_inputs, original_logits)
    print(f"Comparison result: {comparison_result}")
    assert comparison_result, "The comparison failed. Logits are different"