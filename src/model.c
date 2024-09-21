#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"
#include "tokenizer.h"
#include "model.h"

////////////////////////////////////////////////////////// TO TENSORS //////////////////////////////////////////////////////
// Function to convert 2D int array to 1D int64_t array
int64_t* flatten_int_array(int** data, size_t rows, size_t cols) {
    int64_t* flat_data = (int64_t*)malloc(rows * cols * sizeof(int64_t));
    if (!flat_data) {
        fprintf(stderr, "Error: Memory allocation for flat_data failed\n");
        return NULL;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_data[i * cols + j] = (int64_t)data[i][j];
        }
    }
    return flat_data;
}

// Function to create a tensor from data
OrtValue* create_tensor(int64_t* data, size_t rows, size_t cols) {
    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create MemoryInfo: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    int64_t input_dims[2] = { (int64_t)rows, (int64_t)cols };
    OrtValue* tensor = NULL;

    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        data,
        rows * cols * sizeof(int64_t),
        input_dims,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &tensor
    );
    g_ort->ReleaseMemoryInfo(memory_info);

    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create tensor: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    return tensor;
}

// Function for preparing input tensors
int prepare_input_tensors(TokenizedInputs* tokenized, OrtValue** input_ids_tensor, OrtValue** attention_mask_tensor) {
    // preparing input_ids
    int64_t* input_ids_data = flatten_int_array(tokenized->input_ids, tokenized->batch_size, tokenized->seq_length);
    if (input_ids_data == NULL) {
        return -1;
    }
    *input_ids_tensor = create_tensor(input_ids_data, tokenized->batch_size, tokenized->seq_length);
    if (*input_ids_tensor == NULL) {
        free(input_ids_data);
        return -1;
    }

    // preparing attention_mask
    int64_t* attention_mask_data = flatten_int_array(tokenized->attention_mask, tokenized->batch_size, tokenized->seq_length);
    if (attention_mask_data == NULL) {
        free(input_ids_data);
        g_ort->ReleaseValue(*input_ids_tensor);
        return -1;
    }
    *attention_mask_tensor = create_tensor(attention_mask_data, tokenized->batch_size, tokenized->seq_length);
    if (*attention_mask_tensor == NULL) {
        free(input_ids_data);
        free(attention_mask_data);
        g_ort->ReleaseValue(*input_ids_tensor);
        return -1;
    }
    return 0;
}




////////////////////////////////////////////////////// ONNX ////////////////////////////////////////////////////////////////////////
// Function to run model inference
OrtValue* run_inference(OrtSession* session, OrtValue* input_ids_tensor, OrtValue* attention_mask_tensor) {
    OrtStatus* status = NULL;

    // Input node names (they must match your model input nodes)
    const char* input_names[] = { "input_ids", "attention_mask" };
    OrtValue* input_tensors[] = { input_ids_tensor, attention_mask_tensor };

    // Get the number of output nodes
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    if (status != NULL || num_output_nodes == 0) {
        fprintf(stderr, "Error: Failed to get the number of output nodes.\n");
        if (status) g_ort->ReleaseStatus(status);
        return NULL;
    }

    // Get the name of the output node
    OrtAllocator* allocator = NULL;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);

    char* output_name = NULL;
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get the exit node name.\n");
        if (status) g_ort->ReleaseStatus(status);
        return NULL;
    }

    // Run inference
    OrtRunOptions* run_options = NULL; 
    OrtValue* output_tensor = NULL;

    status = g_ort->Run(
        session,
        run_options,
        input_names,
        (const OrtValue* const*)input_tensors,
        2, // Number of input tensors
        (const char* const*)&output_name,
        1, // Number of output nodes
        &output_tensor
    );

    // Free the output node name
    allocator->Free(allocator, output_name);

    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error while performing inference: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    return output_tensor;
}

OrtSession* create_ort_session(OrtEnv* env, const char* model_path) {
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtStatus* status = NULL;

    // Create session options
    status = g_ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create session options: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    #ifdef USE_CUDA // GPU
    int device_id = 0;
    status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to add CUDA Execution Provider: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        return NULL;
    }
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
    printf("\tCUDA Execution Provider added successfully.\n");
    #else
    printf("\tUsing CPU Execution Provider.\n");
    #endif
    

    // Load the model and create a session
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create session: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        return NULL;
    }

    g_ort->ReleaseSessionOptions(session_options);

    return session;
}

OrtEnv* initialize_ort_environment() {
    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "GLiClass", &env);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create env for ONNX Runtime: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }
    return env;
}

void initialize_ort_api() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}