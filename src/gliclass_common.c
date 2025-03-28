#include "GLiClass/gliclass_common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "error.h"

GLiClassStatus* gliclass_create_inference_config(
    size_t batch_size,
    size_t min_length,
    size_t max_length,
    float threshold,
    char* classification_type,
    bool add_prefix_space,
    GLiClassInferenceConfig* config
) {
    char* template = "Inference parameter is invalid: %s";
    if (batch_size == 0) {
        return set_error(GC_LOGICAL_ERROR, template, "batch_size shouldn't equal zero");
    } else if (max_length == 0) {
        return set_error(GC_LOGICAL_ERROR, template, "max_length shouldn't equal zero");
    } else if (min_length > max_length) {
        return set_error(GC_LOGICAL_ERROR, "min_length should be <= max_length");
    } else if (threshold < 0 && threshold > 1) {
        return set_error(GC_LOGICAL_ERROR, "threshold should be in range 0 ... 1");
    } else if (
        strcmp(classification_type, "multi-label") != 0
        && strcmp(classification_type, "single-label") != 0
    ) {
        return set_error(GC_LOGICAL_ERROR, template, "classification_type should be equal to 'multi-label' or 'single-label'");
    }
    config->batch_size = batch_size;
    config->min_length = min_length;
    config->max_length = max_length;
    config->threshold = threshold;
    config->classification_type = classification_type;
    config->add_prefix_space = add_prefix_space;
    return NULL;
}


void gliclass_free_results(GLiClassResult* results, size_t num_results) {
    if (!results) return;
    free(results);
}


void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size) {
    if (!results) return;
    for (size_t i = 0; i < num_results_size; i++) {
        free(results[i]);
    }
    free(results);
    free(num_results);
}

void gliclass_free_status(GLiClassStatus* status) {
    if (!status) return;
    free(status);
}