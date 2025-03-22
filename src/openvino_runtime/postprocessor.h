#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <stdbool.h>
#include "GLiClass/gliclass_ov.h"

void process_output_tensor_openvino(
    GLiClassSessionOpenVino* session,
    const GLiClassInferenceConfig* config,
    ov_tensor_t* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
);

#endif // POSTPROCESSOR_H
