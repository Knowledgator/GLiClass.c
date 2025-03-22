#ifndef UTILS_H
#define UTILS_H

#include "GLiClass/gliclass_common.h"
/**
 * Flattens a 2D array of integers into a 1D array of int64_t for use in tensor creation.
 * 
 * @param data A 2D array of integers.
 * @param rows The number of rows in the 2D array.
 * @param cols The number of columns in the 2D array.
 * @return A pointer to a dynamically allocated 1D int64_t array.
 *         The caller is responsible for freeing the allocated memory.
 */
int64_t* flatten_int_array(int64_t** data, size_t rows, size_t cols);

GLiClassModelConfig* initialize_model_config(const char* model_config_path);

/**
 * Sigmoid function to map logits to probabilities.
 * 
 * @param x The input value (logit).
 * @return The probability corresponding to the logit, calculated using the sigmoid function.
 */
float sigmoid(float x);

void process_multi_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
);

void process_multi_label_batch(
    const float* const output_data,
    const size_t batch_size,
    const size_t num_classes,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size,
    const float threshold,
    const size_t text_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
);

void process_single_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
);

void process_single_label_batch(
    const float* const output_data,
    const size_t batch_size,
    const size_t num_classes,
    const char*** labels,
    const size_t* num_labels,
    const size_t num_labels_size,
    const float threshold,
    const size_t text_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
);

#endif // UTILS_H