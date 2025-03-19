#include "postprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "model.h"
/**
 * Sigmoid function to map logits to probabilities.
 * 
 * @param x The input value (logit).
 * @return The probability corresponding to the logit, calculated using the sigmoid function.
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void process_multi_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    for (size_t j = 0; j < num_classes; j++) {
        float logit = output_data[j];
        float prob = sigmoid(logit);  // sigmoid function

        if (prob < threshold) continue;

        const char* label = NULL;
        if (j < num_labels) {
            label = labels[j];
        }
    
        if (!label) label = "[Unknown]";
        out_results[*out_num_results] = (GLiClassResult){(char*)label, prob};
        *out_num_results += 1;
    }
}

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
) {
    bool same_labels = num_labels_size == 1;
    for (size_t i = 0; i < batch_size; i++) {
        out_results[text_id+i] = (GLiClassResult*)calloc(num_classes, sizeof(GLiClassResult));
        if (!out_results[text_id+i]) {
            fprintf(stderr, "Unable to allocate results");    
        }

        const char** current_labels = NULL;
        size_t current_labels_size;
        if (same_labels) {
            current_labels = labels[0];
            current_labels_size = num_labels[0];
        } else {
            current_labels = labels[i];
            current_labels_size = num_labels[i];
        }
        process_multi_label(
            output_data+i*num_classes, // slide to current batch
            num_classes,
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
}

void process_single_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    float max_prob = 0.0f; // TODO: skip first class
    size_t max_idx = 0;
    for (size_t j = 0; j < num_classes; j++) {
        float logit = output_data[j];
        float prob = sigmoid(logit);  // sigmoid function
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = j;
        }
    }
    if (max_prob < threshold) return;

    const char* label = NULL;
    if (max_idx < num_labels) {
        label = labels[max_idx];
    }

    if (!label) label = "[Unknown]";
    out_results[0] = (GLiClassResult){(char*)label, max_prob};
    *out_num_results += 1;
}

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
) {
    bool same_labels = num_labels_size == 1;
    for (size_t i = 0; i < batch_size; i++) {
        out_results[text_id+i] = (GLiClassResult*)calloc(1, sizeof(GLiClassResult));
        if (!out_results[text_id+i]) {
            fprintf(stderr, "Unable to allocate results");    
        }

        const char** current_labels = NULL;
        size_t current_labels_size;
        if (same_labels) {
            current_labels = labels[0];
            current_labels_size = num_labels[0];
        } else {
            current_labels = labels[i];
            current_labels_size = num_labels[i];
        }
        process_single_label(
            output_data+i*num_classes, // slide to current batch
            num_classes,
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
}

void process_output_tensor(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    OrtValue* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    OrtStatus* status = NULL;

    // Get information about the type and shape of the tensor
    OrtTensorTypeAndShapeInfo* type_info = NULL;
    status = g_ort->GetTensorTypeAndShape(output_tensor, &type_info);
    if (status != NULL) {
        fprintf(stderr, "Error: Unable to obtain information about the tensor type and shape.\n");
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Get the number of dimensions
    size_t num_dims = 0;
    status = g_ort->GetDimensionsCount(type_info, &num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get the number of dimensions of the tensor.\n");
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Get the dimensions of the measurements
    int64_t* dims = (int64_t*)calloc(num_dims, sizeof(int64_t));
    status = g_ort->GetDimensions(type_info, dims, num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get tensor dimension sizes.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Calculate the total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        total_elements *= dims[i];
    }

    // Get a pointer to the tensor data
    float* output_data = NULL;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get tensor data.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    int64_t num_classes = dims[1];
    size_t text_id = 0;
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
            text_id,
            out_results,
            out_num_results
        );
    }

    if (dims) free(dims);
    if (type_info) g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    if (status) g_ort->ReleaseStatus(status);
}

void process_output_tensor_batch(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    OrtValue* output_tensor, 
    const char** labels[],
    const size_t num_labels[],
    const size_t num_labels_size,
    const size_t batch_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
) {
    OrtStatus* status = NULL;

    // Get information about the type and shape of the tensor
    OrtTensorTypeAndShapeInfo* type_info = NULL;
    status = g_ort->GetTensorTypeAndShape(output_tensor, &type_info);
    if (status != NULL) {
        fprintf(stderr, "Error: Unable to obtain information about the tensor type and shape.\n");
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Get the number of dimensions
    size_t num_dims = 0;
    status = g_ort->GetDimensionsCount(type_info, &num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get the number of dimensions of the tensor.\n");
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Get the dimensions of the measurements
    int64_t* dims = (int64_t*)calloc(num_dims, sizeof(int64_t));
    status = g_ort->GetDimensions(type_info, dims, num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get tensor dimension sizes.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Calculate the total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        total_elements *= dims[i];
    }

    // Get a pointer to the tensor data
    float* output_data = NULL;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get tensor data.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) {
            g_ort->ReleaseStatus(status);
        }
        return;
    }

    // Process logits
    int64_t batch_size = dims[0];
    int64_t num_classes = dims[1];
    size_t text_id = batch_id * config->batch_size;
    if (strcmp(config->classification_type, "multi-label") == 0) {    
        process_multi_label_batch(
            output_data,
            batch_size,
            num_classes,
            labels,
            num_labels,
            num_labels_size,
            config->threshold,
            text_id,
            out_results,
            out_num_results
        );
    } else if (strcmp(config->classification_type, "single-label") == 0){
        process_single_label_batch(
            output_data,
            batch_size,
            num_classes,
            labels,
            num_labels,
            num_labels_size,
            config->threshold,
            text_id,
            out_results,
            out_num_results
        );
    }

    if (dims) free(dims);
    if (type_info) g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    if (status) g_ort->ReleaseStatus(status);
}