#include "postprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "../utils.h"
#include "model.h"

void process_output_tensor_openvino(
    GLiClassSessionOpenVino* session,
    const GLiClassInferenceConfig* config,
    ov_tensor_t* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    ov_shape_t output_shape = {0};
    ov_status_e status = ov_tensor_get_shape(output_tensor, &output_shape);
    if (status != OK) return;

    float* output_data = NULL;
    status = ov_tensor_data(output_tensor, (void*)&output_data);
    if (status != OK) {
        ov_shape_free(&output_shape);
        return;
    }

    int64_t num_classes = output_shape.dims[1];
    if (strcmp(config->classification_type, "multi-label") == 0) {    
        process_multi_label(
            output_data,
            num_classes,
            labels,
            num_labels,
            config->threshold,
            out_results,
            out_num_results
        );
    } else if (strcmp(config->classification_type, "single-label") == 0){
        process_single_label(
            output_data,
            num_classes,
            labels,
            num_labels,
            config->threshold,
            out_results,
            out_num_results
        );
    }
    ov_shape_free(&output_shape);
}