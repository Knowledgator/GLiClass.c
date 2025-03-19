#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>
#include "onnxruntime_c_api.h"
#include "GLiClass/gliclass_api.h"
#include "tokenizer.h"

GLiClassModelConfig* initialize_model_config(const char* model_config_path);

///// TO TENSORS /////
int64_t* flatten_int_array(int64_t** data, size_t rows, size_t cols);
OrtValue* create_tensor(int64_t* data, size_t rows, size_t cols);
int prepare_input_tensor(TokenizedInput* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor);
int prepare_input_tensors(TokenizedInputs* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor);

/// ONNX ///
OrtEnv* initialize_ort_environment();
OrtSession* create_ort_session(OrtEnv* env, const char* model_path, int num_threads);
OrtValue* run_inference(OrtSession* session, OrtValue* input_ids_tensor, OrtValue* attention_mask_tensor);

#endif // MODEL_H