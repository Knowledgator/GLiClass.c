#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"
#include "postprocessor.h"

// Function for processing the output tensor (logits)
void process_output_tensor(OrtValue* output_tensor, const OrtApi* g_ort) {
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
    // TODO: add decoding part
    for (size_t i = 0; i < total_elements; ++i) {
        printf("Logit[%zu]: %f\n", i, output_data[i]);
    }

    // Free mem
    free(dims);
    g_ort->ReleaseTensorTypeAndShapeInfo(type_info);
}