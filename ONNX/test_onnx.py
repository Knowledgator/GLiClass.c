import onnxruntime, torch
from transformers import AutoTokenizer
import numpy as np
from gliclass  import GLiClassModel, ZeroShotClassificationPipeline
from tqdm import tqdm

def process_logits(texts, labels, logits, threshold = 0.5, batch_size=8, classification_type = "multi-label"):
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    if isinstance(texts, str):
        texts = [texts]
    if isinstance(labels[0], str):
        same_labels = True
    else:
        same_labels = False
            
    results = []
    iterable = range(0, len(texts), batch_size)
    iterable = tqdm(iterable)


    batch_texts = texts[0:0+batch_size]
    if classification_type == 'single-label':
        for i in range(len(batch_texts)):
            score = torch.softmax(logits[i], dim=-1)
            if same_labels:
                curr_labels = labels
            else:
                curr_labels = labels[i]
            pred_label = curr_labels[torch.argmax(score).item()]
            results.append([{'label': pred_label, 'score': score.max().item()}])

    elif classification_type == 'multi-label':
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)
        for i in range(len(batch_texts)):
            text_results = []
            if same_labels:
                curr_labels = labels
            else:
                curr_labels = labels[i]
            for j, prob in enumerate(probs[i]):
                score = prob.item()
                if score>threshold:
                    try:
                        text_results.append({'label': curr_labels[j], 'score': score})
                    except IndexError:
                        break
            results.append(text_results)

    return results


# Step 1: Loading the ONNX Model
onnx_model_path = "model/multi-label-model_quantized.onnx"#multi-label-model.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Step 2: Tokenize text
model_name = "knowledgator/gliclass-base-v1.0"
gliclass_model = GLiClassModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = ZeroShotClassificationPipeline(gliclass_model, tokenizer, classification_type="multi-label", device="cpu")
print("Loaded")

text = "ONNX is an open-source format designed to enable the interoperability of AI models across various frameworks and tools."
labels = ['format', 'model', 'tool', 'cat']
inputs = pipeline.pipe.prepare_inputs(text, labels)
print("tokenized")

# Step 3: Preparing Input Data for ONNX Model
onnx_inputs = {
    ort_session.get_inputs()[0].name: np.array(inputs["input_ids"]), 
    ort_session.get_inputs()[1].name: np.array(inputs["attention_mask"])
}
print("data prepared")

# Step 4: Run Inference
onnx_outputs = ort_session.run(None, onnx_inputs)
print("Onnx outputs received")
#print(f"ONNX Outputs: {onnx_outputs}")

# Checking the form of logits
logits = onnx_outputs[0]
print(f"Logits shape: {logits.shape}")

if logits.size > 0:
    print("Done! we have logits!")
    print("Decoding logits...")
    res = process_logits(
        texts = text,
        labels = labels,
        logits = logits
    )
    print(res)
else:
    print("Logits are empty, please check the model output.")
