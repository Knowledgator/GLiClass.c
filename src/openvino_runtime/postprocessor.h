#ifndef OPENVINO_POSTPROCESSOR_H
#define OPENVINO_POSTPROCESSOR_H

#include "GLiClass/gliclass_ov.h"

GLiClassStatus openvino_process_output_tensor(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    ov_tensor_t* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
);

#endif // OPENVINO_POSTPROCESSOR_H
