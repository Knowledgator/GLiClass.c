#include "GLiClass/gliclass_common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "error.h"

GLiClassStatus* gliclass_create_inference_config(
    size_t batch_size,
    float threshold,
    char* classification_type,
    GLiClassInferenceConfig* config
) {
    char* template = "Inference parameter is invalid: %s";
    if (batch_size == 0) {
        return set_error(GC_LOGICAL_ERROR, template, "batch_size shouldn't equal zero");
    } else if (threshold < 0 && threshold > 1) {
        return set_error(GC_LOGICAL_ERROR, "threshold should be in range 0 ... 1");
    } else if (
        strcmp(classification_type, "multi-label") != 0
        && strcmp(classification_type, "single-label") != 0
    ) {
        return set_error(GC_LOGICAL_ERROR, template, "classification_type should be equal to 'multi-label' or 'single-label'");
    }
    config->batch_size = batch_size;
    config->threshold = threshold;
    config->classification_type = classification_type;
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