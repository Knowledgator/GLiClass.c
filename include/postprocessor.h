#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "gliclass_api.h"

float sigmoid(float x);
void process_output_tensor(
    GLiClassSession* session,
    OrtValue* output_tensor, 
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size,
    const size_t batch_id,
    GLiClassResult** out_results[],
    size_t out_num_results[]
);

#endif // POSTPROCESSOR_H
