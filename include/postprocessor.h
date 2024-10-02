#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <stdbool.h>
#include "onnxruntime_c_api.h"

float sigmoid(float x);
void process_output_tensor(OrtValue* output_tensor, const OrtApi* g_ort, bool same_labels, const char** const* labels,
                            const size_t* num_labels, size_t num_labels_size, float threshold, size_t num_texts, const char** texts,
                            const char* classification_type);

#endif // POSTPROCESSOR_H
