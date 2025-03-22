#include "GLiClass/gliclass_ov.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "preprocessor.h"
#include "utils.h"
#include "openvino_runtime/model.h"
#include "openvino_runtime/postprocessor.h"

#ifndef _WIN32
    #include <unistd.h>
    #include <mqueue.h>
    #include <pthread.h>
#else
    #include <io.h>

    #define access _access
    #define F_OK 0
#endif

// Mutex declarations
#ifndef _WIN32
pthread_mutex_t queue_mutex;
#else
HANDLE queue_mutex;
#endif

GLiClassSessionOpenVino* gliclass_init_openvino_runtime(
    const char* model_path,
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const char* device_type,
    const bool use_mutex
) {
    GLiClassSessionOpenVino* session = (GLiClassSessionOpenVino*)calloc(1, sizeof(GLiClassSessionOpenVino));
    if (!session) {
        fprintf(stderr, "Unable to allocate session");
        return NULL;
    }

    session->use_mutex = use_mutex;
    if (session->use_mutex) {
        // Initialize queue mutex
        #ifndef _WIN32
        pthread_mutex_init(&queue_mutex, NULL);
        #else
        queue_mutex = CreateMutex(NULL, FALSE, NULL);
        #endif
    }

    // Initialize the model config
    session->model_config = initialize_model_config(model_config_path);
    if (!session->model_config) {
        fprintf(stderr, "Unable to init model config\n");
        gliclass_cleanup_openvino(session);
        return NULL;
    }

    // Initialize tokenizer
    session->tokenizer = create_tokenizer(tokenizer_path);
    if (!session->tokenizer) {
        fprintf(stderr, "Unable to init tokenizer\n");
        gliclass_cleanup_openvino(session);
        return NULL;
    }

    session->core = NULL;
    ov_status_e status = ov_core_create(&session->core);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to init OpenVino core: %s", e);
        gliclass_cleanup_openvino(session);
        return NULL;
    }

    
    char threads[3];
    snprintf(threads, 3, "%d", num_threads);
    session->model = NULL;
    if (num_threads > 0 && strcmp(device_type, "CPU") != 0) {
        status = ov_core_compile_model_from_file(
            session->core, model_path, device_type, 4, &session->model,
            ov_property_key_hint_performance_mode, "LATENCY",
            ov_property_key_inference_num_threads, threads
        );
    } else {
        status = ov_core_compile_model_from_file(
            session->core, model_path, device_type, 2, &session->model,
            ov_property_key_hint_performance_mode, "LATENCY"
        );
    }

    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to compile model: %s: %s\n", e, ov_get_last_err_msg());
        gliclass_cleanup_openvino(session);
        return NULL;
    }
    return session;
}


bool gliclass_infer_openvino(
    GLiClassSessionOpenVino* session,
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results,
    bool* truncated
) {
    if (!session || !input_text || !labels || num_labels == 0) {
        fprintf(stderr, "Inputs have invalid value!\n");
        return false;
    }

    // Allocate output array for results
    *out_num_results = 0;
    *out_results = (GLiClassResult*)calloc(num_labels, sizeof(GLiClassResult));
    if (!*out_results) {
        fprintf(stderr, "Unable to allocate results\n");
        return false;
    }

    char* input = prepare_input(input_text, labels, num_labels, session->model_config->prompt_first, config->add_prefix_space);
    if (!input) {
        fprintf(stderr, "Error while preparing text\n");
        return false;
    }

    TokenizedInput tokenized = tokenize_input(
        session->tokenizer, 
        (const char*)input, 
        config->max_length
    );
    *truncated = tokenized.truncated;

    ov_tensor_t* input_ids_tensor = NULL;
    ov_tensor_t* attention_mask_tensor = NULL;
    prepare_input_tensor_openvino(
        &tokenized,
        &input_ids_tensor, 
        &attention_mask_tensor
    );

    ov_tensor_t* output_tensor = NULL;
    if (session->use_mutex) {
        #ifndef _WIN32
        pthread_mutex_lock(&queue_mutex);
        #else
        WaitForSingleObject(queue_mutex, INFINITE); 
        #endif
        output_tensor = run_inference_openvino(session->model, input_ids_tensor, attention_mask_tensor);
        #ifndef _WIN32
        pthread_mutex_unlock(&queue_mutex);
        #else
        ReleaseMutex(queue_mutex);
        #endif
    } else {
        output_tensor = run_inference_openvino(session->model, input_ids_tensor, attention_mask_tensor);
    }
    ov_tensor_free(input_ids_tensor);
    ov_tensor_free(attention_mask_tensor);
    free_tokenized_input(&tokenized);
    free(input);
    
    process_output_tensor_openvino(
        session,
        config,
        output_tensor, 
        labels, 
        num_labels, 
        *out_results,
        out_num_results
    );
    ov_tensor_free(output_tensor);
    return true;
}


void gliclass_cleanup_openvino(GLiClassSessionOpenVino* session) {
    if (!session) return;
    if (session->use_mutex) {
        #ifndef _WIN32
        pthread_mutex_destroy(&queue_mutex);
        #else
        CloseHandle(queue_mutex);
        #endif
    }
    if (session->model_config) free((void*)session->model_config);
    if (session->tokenizer) tokenizers_free(session->tokenizer);
    if (session->model) ov_compiled_model_free(session->model);
    if (session->core) ov_core_free(session->core);
    free(session);
}