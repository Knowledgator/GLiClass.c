#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include "GLiClass/gliclass_ov.h"
#include "../tokenizer.h"

int prepare_input_tensor_openvino(
    TokenizedInput* tokenized, ov_tensor_t** input_ids_tensor, ov_tensor_t** attention_mask_tensor
);

/**
 * Runs inference using the ONNX model session and input tensors.
 * 
 * @param model A pointer to the OpenVino compiled model
 * @param input_ids_tensor A pointer to the OrtValue representing the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue representing the attention mask tensor.
 * @return A pointer to an OrtValue containing the model's output, or NULL if inference fails.
 * IMPORTANT: The caller is responsible for releasing the output_tensor via g_ort->ReleaseValue(output_tensor)
 */
ov_tensor_t* run_inference_openvino(
    ov_compiled_model_t* model, ov_tensor_t* input_ids_tensor, ov_tensor_t* attention_mask_tensor
);

#endif // MODEL_H