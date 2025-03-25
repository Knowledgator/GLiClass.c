#ifndef OPENVINO_POSTPROCESSOR_H
#define OPENVINO_POSTPROCESSOR_H

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

#endif // OPENVINO_POSTPROCESSOR_H
