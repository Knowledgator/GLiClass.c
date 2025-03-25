#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include <stddef.h>
#include "GLiClass/gliclass_ort.h"
#include "../tokenizer.h"

/**
 * Prepares input tensors for the ONNX model using tokenized input data.
 * 
 * @param tokenized A pointer to the TokenizedInputs structure containing the tokenized data.
 * @param input_ids_tensor A pointer to the OrtValue that will store the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue that will store the attention mask tensor.
 * @return 0 if successful, -1 if an error occurs during tensor preparation.
 */
int prepare_input_tensor(TokenizedInput* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor);

/**
 * Prepares input tensors for the ONNX model using tokenized input data.
 * 
 * @param tokenized A pointer to the TokenizedInputs structure containing the tokenized data.
 * @param input_ids_tensor A pointer to the OrtValue that will store the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue that will store the attention mask tensor.
 * @return 0 if successful, -1 if an error occurs during tensor preparation.
 */
int prepare_input_tensors(TokenizedInputs* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor);

/**
 * Runs inference using the ONNX model session and input tensors.
 * 
 * @param session A pointer to the ONNX model session.
 * @param input_ids_tensor A pointer to the OrtValue representing the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue representing the attention mask tensor.
 * @return A pointer to an OrtValue containing the model's output, or NULL if inference fails.
 * IMPORTANT: The caller is responsible for releasing the output_tensor via g_ort->ReleaseValue(output_tensor)
 */
OrtValue* run_inference(OrtSession* session, OrtValue* input_ids_tensor, OrtValue* attention_mask_tensor);

#endif // ONNX_MODEL_H