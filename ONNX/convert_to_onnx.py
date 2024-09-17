import os
import argparse
import numpy as np

from gliclass  import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

import torch, json
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_original_logits(model, tokenized_inputs) -> list:
    with torch.no_grad():
        model_output = model(**tokenized_inputs)
    logits = model_output.logits       
    logits = logits.round(decimals=4)

    return logits.tolist()

def create_config(original_model_name, architecture_type, original_logits, save_path) -> None:
    data = {
        "original_model_name" :original_model_name,
        "architecture_type" : architecture_type,
        "original_logits" : original_logits
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "knowledgator/gliclass-base-v1.0")
    parser.add_argument('--save_path', type=str, default = 'model/')
    parser.add_argument('--quantize', type=bool, default = True)
    parser.add_argument('--classification_type', type=str, default = "multi-label")

    args = parser.parse_args()
    
    if args.classification_type not in ['single-label', "multi-label"]:
        raise NotImplementedError("This type is not supported yet")

    os.makedirs(args.save_path, exist_ok= True)
    
    onnx_save_path = os.path.join(args.save_path, "model.onnx")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading a model...")
    gliclass_model = GLiClassModel.from_pretrained(args.model_path)
    architecture_type = gliclass_model.config.architecture_type
    if architecture_type != 'uni-encoder':
        raise NotImplementedError("This artchitecture is not implemented for ONNX yet")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pipeline = ZeroShotClassificationPipeline(gliclass_model, tokenizer, classification_type=args.classification_type, device=device)

    text = "ONNX is an open-source format designed to enable the interoperability of AI models across various frameworks and tools."
    labels = ['format', 'model', 'tool', 'cat']

    tokenized_inputs = pipeline.pipe.prepare_inputs(text, labels)

    all_inputs = (tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
    input_names = ['input_ids', 'attention_mask']
    dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "position", 1: "batch_size"}
        }

    print("Converting...")
    torch.onnx.export(
        gliclass_model,             # Model
        all_inputs,                 # Inputs for exprt
        onnx_save_path,             # output file name
        input_names=input_names,    # Output data name
        output_names=["logits"],    # output logits names
        dynamic_axes=dynamic_axes,  # Dynamic Axes
        opset_version=14
    )

    if args.quantize:
        quantized_save_path = os.path.join(args.save_path, "model-quantized.onnx")
        # Quantize the ONNX model
        print("Quantizing the model...")
        quantize_dynamic(
            onnx_save_path,  # Input model
            quantized_save_path,  # Output model
            weight_type=QuantType.QUInt8  # Quantize weights to 8-bit integers
        )
    print("Creating configuration file...")
    config_path = args.save_path + "config.json"
    create_config(
        original_model_name = args.model_path,
        architecture_type = architecture_type,
        original_logits = get_original_logits(gliclass_model, tokenized_inputs),
        save_path = config_path
    )

    print("Done")