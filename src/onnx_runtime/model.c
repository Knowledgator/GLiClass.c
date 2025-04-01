#include "model.h"

#include <stdio.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

#include "../utils.h"
#include "../error.h"

/**
 * Creates a tensor from flattened data.
 * 
 * @param data A 1D array of int64_t representing the flattened tensor data.
 * @param rows The number of rows in the tensor.
 * @param cols The number of columns in the tensor.
 * @return A pointer to an OrtValue representing the tensor, or NULL if tensor creation fails.
 */
GLiClassStatus* ort_create_tensor(int64_t* data, size_t rows, size_t cols, OrtValue** tensor) {
    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    GLiClassStatus* gc_status = NULL;
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Error: Failed to create MemoryInfo: %s", msg);
        g_ort->ReleaseStatus(status);
        return gc_status;
    }

    int64_t input_dims[2] = { (int64_t)rows, (int64_t)cols };

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        data,
        rows * cols * sizeof(int64_t),
        input_dims,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        tensor
    );
    g_ort->ReleaseMemoryInfo(memory_info);

    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Error: Failed to create tensor: %s", msg);
        g_ort->ReleaseStatus(status);
        return gc_status;
    }

    return NULL;
}


GLiClassStatus* ort_prepare_input_tensors(
    const TokenizedInput* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor
) {
    GLiClassStatus* status = ort_create_tensor(
        tokenized->input_ids, 1, tokenized->seq_length, input_ids_tensor
    );
    if (status != NULL) return status;

    status = ort_create_tensor(tokenized->attention_mask, 1, tokenized->seq_length, attention_mask_tensor);
    if (status != NULL) {
        g_ort->ReleaseValue(*input_ids_tensor);
        return status;
    }
    return NULL;
}


GLiClassStatus* ort_prepare_input_tensors_batch(
    const TokenizedInputs* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor
) {
    // preparing input_ids
    int64_t* input_ids_data = NULL;
    GLiClassStatus* status = flatten_int_array(
        tokenized->input_ids, tokenized->batch_size, tokenized->seq_length, &input_ids_data
    );
    if (status != NULL) return status;
    
    status = ort_create_tensor(input_ids_data, tokenized->batch_size, tokenized->seq_length, input_ids_tensor);
    if (status != NULL) {
        free(input_ids_data);
        return status;
    }

    // preparing attention_mask
    int64_t* attention_mask_data = NULL;
    status = flatten_int_array(
        tokenized->attention_mask, tokenized->batch_size, tokenized->seq_length, &attention_mask_data
    );
    if (status != NULL) {
        free(input_ids_data);
        g_ort->ReleaseValue(*input_ids_tensor);
        return status;
    }

    status = ort_create_tensor(attention_mask_data, tokenized->batch_size, tokenized->seq_length, attention_mask_tensor);
    if (status != NULL) {
        free(input_ids_data);
        free(attention_mask_data);
        g_ort->ReleaseValue(*input_ids_tensor);
        return status;
    }
    return status;
}


GLiClassStatus* ort_run_inference(
    GLiClassORTSession* session, OrtValue* input_ids_tensor, OrtValue* attention_mask_tensor, OrtValue** output_tensor
) {
    OrtStatus* status = NULL;
    OrtRunOptions* run_options = NULL;
    OrtAllocator* allocator = NULL;
    char* output_name = NULL;
    GLiClassStatus* gc_status = NULL;

    // Create options to run inference
    status = g_ort->CreateRunOptions(&run_options);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Failed to create run options: %s", msg);
        g_ort->ReleaseStatus(status);
        return gc_status;
    }

    // Get the default allocator
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Failed to get allocator: %s", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return gc_status;
    }

    // Get the number of output nodes
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(session->session, &num_output_nodes);
    if (status != NULL || num_output_nodes == 0) {
        if (status) g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return set_error(GC_INFERENCE_ERROR, "Failed to get output nodes count or no output nodes found");
    }

    // Get the name of the output node
    status = g_ort->SessionGetOutputName(session->session, 0, allocator, &output_name);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Failed to get output name: %s", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return gc_status;
    }

    // Set up input parameters
    const char* const input_names[] = { "input_ids", "attention_mask" };
    const char* const output_names[] = { output_name };
    const OrtValue* const input_tensors[] = { input_ids_tensor, attention_mask_tensor };

    // Run inference
    status = g_ort->Run(
        session->session,
        run_options,
        input_names,
        input_tensors,
        2,  // number of input tensors
        output_names,
        1,  // number of output tensors
        output_tensor
    );

    // Free the memory of the output name
    if (output_name) {
        allocator->Free(allocator, output_name);
    }

    // Free up run options, they are no longer needed
    g_ort->ReleaseRunOptions(run_options);

    // Check the result of the inference
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        gc_status = set_error(GC_INFERENCE_ERROR, "Error during inference: %s\n", msg);
        g_ort->ReleaseStatus(status);
        if (*output_tensor) {
            g_ort->ReleaseValue(*output_tensor);
        }
        return gc_status;
    }
    return NULL;
}