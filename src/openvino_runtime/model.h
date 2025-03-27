#ifndef OPENVINO_MODEL_H
#define OPENVINO_MODEL_H

#include "GLiClass/gliclass_ov.h"
#include "../tokenizer.h"

GLiClassStatus openvino_prepare_input_tensors(
    const TokenizedInput* tokenized, ov_tensor_t** input_ids_tensor, ov_tensor_t** attention_mask_tensor
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
GLiClassStatus openvino_run_inference(
    GLiClassOpenVinoSession* session, 
    ov_tensor_t* input_ids_tensor, 
    ov_tensor_t* attention_mask_tensor, 
    ov_tensor_t** output_tensor
);

#endif // OPENVINO_MODEL_H