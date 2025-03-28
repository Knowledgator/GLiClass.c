#include "postprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../utils.h"
#include "../error.h"
#include "model.h"

GLiClassStatus ort_process_output_tensor(
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
        set_error("Unable to obtain information about the tensor type and shape.");
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get the number of dimensions
    size_t num_dims = 0;
    status = g_ort->GetDimensionsCount(type_info, &num_dims);
    if (status) {
        set_error("Failed to get the number of dimensions of the tensor.");
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get the dimensions of the measurements
    int64_t* dims = (int64_t*)calloc(num_dims, sizeof(int64_t));
    if (!dims) {
        set_error("Unable to allocate dimensions.");
        return GC_MEMORY_ERROR;
    }

    status = g_ort->GetDimensions(type_info, dims, num_dims);
    if (status) {
        set_error("Failed to get tensor dimension sizes.");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get a pointer to the tensor data
    float* output_data = NULL;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status) {
        fprintf(stderr, "Error: Failed to get tensor data.\n");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    int64_t num_classes = dims[1];
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

    free(dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    return GC_OK;
}


GLiClassStatus ort_process_output_tensor_batch(
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
        set_error("Unable to obtain information about the tensor type and shape.");
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get the number of dimensions
    size_t num_dims = 0;
    status = g_ort->GetDimensionsCount(type_info, &num_dims);
    if (status != NULL) {
        set_error("Failed to get the number of dimensions of the tensor.");
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get the dimensions of the measurements
    int64_t* dims = (int64_t*)calloc(num_dims, sizeof(int64_t));
    status = g_ort->GetDimensions(type_info, dims, num_dims);
    if (status != NULL) {
        set_error("Failed to get tensor dimension sizes.");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Get a pointer to the tensor data
    float* output_data = NULL;
    status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        set_error("Failed to get tensor data.");
        free(dims);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
        g_ort->ReleaseStatus(status);
        return GC_INFERENCE_ERROR;
    }

    // Process logits
    int64_t batch_size = dims[0];
    int64_t num_classes = dims[1];
    size_t text_id = batch_id * config->batch_size;
    GLiClassStatus gc_status = GC_OK;
    if (strcmp(config->classification_type, "multi-label") == 0) {    
        gc_status = process_multi_label_batch(
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
        gc_status = process_single_label_batch(
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

    free(dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
    return gc_status;
}