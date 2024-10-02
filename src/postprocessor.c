#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "onnxruntime_c_api.h"
#include "postprocessor.h"

/**
 * Sigmoid function to map logits to probabilities.
 * 
 * @param x The input value (logit).
 * @return The probability corresponding to the logit, calculated using the sigmoid function.
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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
void process_output_tensor(OrtValue* output_tensor, const OrtApi* g_ort, bool same_labels, char*** labels,
                            size_t* num_labels, size_t num_labels_size, float threshold, size_t num_texts, char** texts,
                            char* classification_type) {
    OrtStatus* status = NULL;

    // Get information about the type and shape of the tensor
    OrtTensorTypeAndShapeInfo* type_info = NULL;
    status = g_ort->GetTensorTypeAndShape(output_tensor, &type_info);
    if (status != NULL) {
        fprintf(stderr, "Error: Unable to obtain information about the tensor type and shape.\n");
        if (status) g_ort->ReleaseStatus(status);
        return;
    }

    // Get the number of dimensions
    size_t num_dims = 0;
    status = g_ort->GetDimensionsCount(type_info, &num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get the number of dimensions of the tensor.\n");
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) g_ort->ReleaseStatus(status);
        return;
    }

    // Get the dimensions of the measurements
    int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
    status = g_ort->GetDimensions(type_info, dims, num_dims);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get tensor dimension sizes.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        if (status) g_ort->ReleaseStatus(status);
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
        if (status) g_ort->ReleaseStatus(status);
        return;
    }


    // Process logits
    int batch_size = dims[0];
    int num_classes = dims[1];
    if (strcmp(classification_type, "multi-label") == 0) {    
        for (int i = 0; i < batch_size; i++) {
            printf("Text: %s:\n", texts[i]);
            for (int j = 0; j < num_classes; j++) {
                float logit = output_data[i * num_classes + j];
                float prob = sigmoid(logit);  // sigmoid function
                
                if (prob > threshold) {
                    const char* label = NULL;
                    if (same_labels) {
                        if (j < num_labels_size) {
                            label = labels[0][j];
                        }
                    } else {
                        if (i < num_texts   && j < num_labels[i]) {
                            label = labels[i][j];
                        }
                    }
                
                    if (label) {
                        printf("  Label: %s, Score: %.6f\n", label, prob);
                    } else {
                        printf("  Label: [Unknown], Score: %.6f\n", prob);
                    }
                }
            }
            printf("\n");
        }
    } else if (strcmp(classification_type, "single-label") == 0){
        for (int i = 0; i < batch_size; i++) {
            printf("Text: %s:\n", texts[i]);
            float max_prob = 0.0f;
            int max_idx = -1;
            for (int j = 0; j < num_classes; j++) {
                float logit = output_data[i * num_classes + j];
                float prob = sigmoid(logit);  // sigmoid function
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = j;
                }
            }
            
            const char* label = NULL;
            if (same_labels) {
                if (max_idx < num_labels_size) {
                    label = labels[0][max_idx];
                }
            } else {
                if (i < num_texts && max_idx < num_labels[i]) {
                    label = labels[i][max_idx];
                }
            }
            
            if (label) {
                printf("  Label: %s, Score: %.6f\n", label, max_prob);
            } else {
                printf("  Label: [Unknown], Score: %.6f\n", max_prob);
            }
            printf("\n");
        }
    }else{
        printf("This type of classification is not supported\n");
    }


    if (dims) free(dims);
    if (type_info) g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    if (status) g_ort->ReleaseStatus(status);
}