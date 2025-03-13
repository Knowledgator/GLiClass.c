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
    const size_t batch_size,
    const size_t num_classes,
    const char*** labels,
    const size_t* num_labels,
    const size_t num_labels_size,
    const float threshold,
    const size_t text_id,
    GLiClassResult** out_results[],
    size_t out_num_results[]
) {
    for (size_t i = 0; i < batch_size; i++) {
        out_results[text_id+i] = (GLiClassResult**)calloc(num_classes, sizeof(GLiClassResult*));
        if (!out_results[text_id+i]) {
            fprintf(stderr, "Unable to allocate results");    
        }

        fprintf(stdout, "TEXT: %ld\n", i); 
        for (size_t j = 0; j < num_classes; j++) {
            float logit = output_data[i * num_classes + j];
            float prob = sigmoid(logit);  // sigmoid function
            
            fprintf(stdout, "%ld : %f\n", j, prob);
            if (prob < threshold) continue;

            const char* label = NULL;
            if (num_labels_size == 1) {
                if (j < num_labels[0]) {
                    label = labels[0][j];
                }
            } else if (i < num_labels_size && j < num_labels[i]) {
                label = labels[i][j];
            }
        
            out_num_results[text_id+i] += 1;
            if (!label) label = "[Unknown]";
            out_results[text_id+i][j] = (GLiClassResult*)calloc(1, sizeof(GLiClassResult));
            *out_results[text_id+i][j] = (GLiClassResult){(char*)label, prob};
        }
    }
}

void process_single_label(
    const float* const output_data,
    const size_t batch_size,
    const size_t num_classes,
    const char*** labels,
    const size_t* num_labels,
    const size_t num_labels_size,
    const size_t text_id,
    GLiClassResult** out_results[],
    size_t out_num_results[]
) {
    for (size_t i = 0; i < batch_size; i++) {
        out_results[text_id+i] = (GLiClassResult**)calloc(1, sizeof(GLiClassResult*));
        if (!out_results[text_id+i]) {
            fprintf(stderr, "Unable to allocate results");    
        }
        out_num_results[text_id+i] = 1;

        float max_prob = 0.0f; // TODO: skip first class
        size_t max_idx = 0;
        for (size_t j = 0; j < num_classes; j++) {
            float logit = output_data[i * num_classes + j];
            float prob = sigmoid(logit);  // sigmoid function
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = j;
            }
        }
        
        const char* label = NULL;
        if (num_labels_size == 1) {
            if (max_idx < num_labels[0]) {
                label = labels[0][max_idx];
            }
        } else if (i < num_labels_size && max_idx < num_labels[i]) {
            label = labels[i][max_idx];
        }
    
        if (!label) label = "[Unknown]";

        out_results[text_id+i][0] = (GLiClassResult*)calloc(1, sizeof(GLiClassResult));
        *out_results[text_id+i][0] = (GLiClassResult){(char*)label, max_prob};
    }
}

/**
 * Processes the output tensor (logits) and prints the predicted labels and scores based on the given classification type (multi-label or single-label).
 * 
 * @param output_tensor A pointer to the OrtValue containing the output logits from the ONNX model.
 * @param g_ort A pointer to the ONNX Runtime API.
 * @param same_labels Boolean indicating if all texts share the same set of labels.
 * @param labels A 2D array of strings containing the labels for each class.
 * @param num_labels A dynamic array indicating the number of labels for each text.
 * @param num_labels_size The number of labels if all texts share the same set.
 * @param threshold The probability threshold for multi-label classification.
 * @param num_texts The number of texts in the batch.
 * @param texts A dynamic array containing the input texts.
 * @param classification_type A string specifying the type of classification ("multi-label" or "single-label").
 */
void process_output_tensor(
    GLiClassSession* session,
    OrtValue* output_tensor, 
    const char** labels[],
    const size_t num_labels[],
    const size_t num_labels_size,
    const size_t batch_id,
    GLiClassResult** out_results[],
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
    size_t text_id = batch_id * session->inference_config->batch_size;
    if (strcmp(session->inference_config->classification_type, "multi-label") == 0) {    
        process_multi_label(
            output_data,
            batch_size,
            num_classes,
            labels,
            num_labels,
            num_labels_size,
            session->inference_config->threshold,
            text_id,
            out_results,
            out_num_results
        );
    } else if (strcmp(session->inference_config->classification_type, "single-label") == 0){
        process_single_label(
            output_data,
            batch_size,
            num_classes,
            labels,
            num_labels,
            num_labels_size,
            text_id,
            out_results,
            out_num_results
        );
    }

    if (dims) free(dims);
    if (type_info) g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    if (status) g_ort->ReleaseStatus(status);
}