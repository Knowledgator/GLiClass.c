import onnxruntime
import json
import torch, os
import numpy as np
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from typing import Tuple, List
import math

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

def run_inference(ort_session, onnx_inputs, labels):
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_tensor = torch.tensor(onnx_outputs[0])
    for j, label in enumerate(labels):
        logit = onnx_tensor[j]
        prob = 1 / (1 + math.exp(-logit.argmax()))

        yield {
            "label": label, 
            "score": prob
        }

if __name__ == "__main__":
    original_model_name, architecture_type, original_logits = load_config("./onnx/config.json")
    original_logits = torch.Tensor(original_logits)

    # Step 1: Loading the ONNX Model
    onnx_model_path = "./onnx/model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Step 2: Tokenize text
    gliclass_model = GLiClassModel.from_pretrained(original_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name, add_prefix_space=True)
    pipeline = ZeroShotClassificationPipeline(gliclass_model, tokenizer, classification_type="multi-label", device="cpu")
    print(f"Model {original_model_name} loaded")

    with open("test_ascii.txt", "r") as f:
        texts = f.readlines() 
    labels = ["format", "model", "tool", "necessity", "person", "city", "location", "country", "war"]

    # Step 4: Run Inference and compare logits
    with open("scores_python_onnx.txt", "w") as f:
        for text in texts:
            onnx_inputs = prepare_onnx_inputs(pipeline, text, labels)
            for i, result in enumerate(run_inference(ort_session, onnx_inputs, labels)):
                f.write(f"Label_{i}: {result['label']}, score: {result['score']}\n")