#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"
#include "tokenizer.h"
#include "model.h"

////////////////////////////////////////////////////////// TO TENSORS //////////////////////////////////////////////////////
/**
 * Flattens a 2D array of integers into a 1D array of int64_t for use in tensor creation.
 * 
 * @param data A 2D array of integers.
 * @param rows The number of rows in the 2D array.
 * @param cols The number of columns in the 2D array.
 * @return A pointer to a dynamically allocated 1D int64_t array.
 *         The caller is responsible for freeing the allocated memory.
 */
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

/**
 * Creates a tensor from flattened data.
 * 
 * @param data A 1D array of int64_t representing the flattened tensor data.
 * @param rows The number of rows in the tensor.
 * @param cols The number of columns in the tensor.
 * @return A pointer to an OrtValue representing the tensor, or NULL if tensor creation fails.
 */
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

/**
 * Prepares input tensors for the ONNX model using tokenized input data.
 * 
 * @param tokenized A pointer to the TokenizedInputs structure containing the tokenized data.
 * @param input_ids_tensor A pointer to the OrtValue that will store the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue that will store the attention mask tensor.
 * @return 0 if successful, -1 if an error occurs during tensor preparation.
 */
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
/**
 * Runs inference using the ONNX model session and input tensors.
 * 
 * @param session A pointer to the ONNX model session.
 * @param input_ids_tensor A pointer to the OrtValue representing the input IDs tensor.
 * @param attention_mask_tensor A pointer to the OrtValue representing the attention mask tensor.
 * @return A pointer to an OrtValue containing the model's output, or NULL if inference fails.
 */
OrtValue* run_inference(OrtSession* session, OrtValue* input_ids_tensor, OrtValue* attention_mask_tensor) {
    OrtStatus* status = NULL;
    OrtRunOptions* run_options = NULL;
    OrtValue* output_tensor = NULL;
    OrtAllocator* allocator = NULL;
    char* output_name = NULL;
    
    // Create options to run inference
    status = g_ort->CreateRunOptions(&run_options);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create run options: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    // Get the default allocator
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to get allocator: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return NULL;
    }

    // Get the number of output nodes
    size_t num_output_nodes = 0;
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    if (status != NULL || num_output_nodes == 0) {
        fprintf(stderr, "Error: Failed to get output nodes count or no output nodes found\n");
        if (status) g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return NULL;
    }

    // Get the name of the output node
    status = g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    if (status != NULL) {
        fprintf(stderr, "Error: Failed to get output name\n");
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseRunOptions(run_options);
        return NULL;
    }

    // Set up input parameters
    const char* input_names[] = { "input_ids", "attention_mask" };
    const char* output_names[] = { output_name };
    OrtValue* input_tensors[] = { input_ids_tensor, attention_mask_tensor };

    // Run inference
    status = g_ort->Run(
        session,
        run_options,
        input_names,
        (const OrtValue* const*)input_tensors,
        2,  // number of input tensors
        (const char* const*)output_names,
        1,  // number of output tensors
        &output_tensor
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
        fprintf(stderr, "Error during inference: %s\n", msg);
        g_ort->ReleaseStatus(status);
        if (output_tensor) {
            g_ort->ReleaseValue(output_tensor);
        }
        return NULL;
    }
    
    // Return the result
    // IMPORTANT: The caller is responsible for releasing the output_tensor
    // via g_ort->ReleaseValue(output_tensor)
    return output_tensor;
}

/**
 * Creates and initializes an ONNX Runtime session from a model file.
 * 
 * @param env A pointer to the ONNX Runtime environment.
 * @param model_path The file path to the ONNX model.
 * @param num_threads The number of threads to use for inference (CPU only).
 * @return A pointer to the OrtSession if successful, or NULL if an error occurs.
 */
OrtSession* create_ort_session(OrtEnv* env, const char* model_path, int num_threads) {
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

    // Set the number of threads for intra-op operations
    status = g_ort->SetIntraOpNumThreads(session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to set intra-op threads: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        return NULL;
    }

    // Set the number of threads for inter-op operations
    status = g_ort->SetInterOpNumThreads(session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to set inter-op threads: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
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

/**
 * Initializes the ONNX Runtime environment.
 * 
 * @return A pointer to the OrtEnv, or NULL if environment creation fails.
 */
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

/**
 * Initializes the ONNX Runtime API.
 */
void initialize_ort_api() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
}