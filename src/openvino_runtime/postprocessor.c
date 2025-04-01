#include "postprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <openvino/c/openvino.h>

#include "../utils.h"
#include "../error.h"
#include "model.h"

GLiClassStatus* openvino_process_output_tensor(
    const GLiClassInferenceConfig* config,
    ov_tensor_t* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    ov_shape_t output_shape = {0};
    ov_status_e status = ov_tensor_get_shape(output_tensor, &output_shape);
    GLiClassStatus* gc_status = NULL;
    if (status != OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        gc_status = set_error(GC_INFERENCE_ERROR, "Unable to get output tensor shape: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        return gc_status;
    }

    float* output_data = NULL;
    status = ov_tensor_data(output_tensor, (void*)&output_data);
    if (status != OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        gc_status = set_error(GC_INFERENCE_ERROR, "Unable to get output tensor data: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        ov_shape_free(&output_shape);
        return gc_status;
    }

    size_t num_classes = num_labels > (size_t)output_shape.dims[1] ? (size_t)output_shape.dims[1] : num_labels;
    if (strcmp(config->classification_type, "multi-label") == 0) {    
        process_multi_label(
            output_data,
            labels,
            num_classes,
            config->threshold,
            out_results,
            out_num_results
        );
    } else if (strcmp(config->classification_type, "single-label") == 0){
        process_single_label(
            output_data,
            labels,
            num_classes,
            config->threshold,
            out_results,
            out_num_results
        );
    }
    ov_shape_free(&output_shape);
    return NULL;
}