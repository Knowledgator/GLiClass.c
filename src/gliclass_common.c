#include "GLiClass/gliclass_common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

bool gliclass_create_inference_config(
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
        fprintf(stderr, template, "batch_size shouldn't equal zero");
        return false;
    } else if (max_length == 0) {
        fprintf(stderr, template, "max_length shouldn't equal zero");
        return false;
    } else if (min_length > max_length) {
        fprintf(stderr, template, "min_length should be <= max_length");
        return false;
    } else if (threshold < 0 && threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (threshold < 0 && threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (
        strcmp(classification_type, "multi-label") != 0
        && strcmp(classification_type, "single-label") != 0
    ) {
        fprintf(stderr, template, "classification_type should be equal to 'multi-label' or 'single-label'");
        return false;
    }
    config->batch_size = batch_size;
    config->min_length = min_length;
    config->max_length = max_length;
    config->threshold = threshold;
    config->classification_type = classification_type;
    config->add_prefix_space = add_prefix_space;
    return true;
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