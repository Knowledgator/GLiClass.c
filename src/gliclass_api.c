#include "GLiClass/gliclass_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "model.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "parallel_processor.h"

#ifndef _WIN32
    #include <mqueue.h>
    #include <pthread.h>
#else
    #include <windows.h>
#endif

#ifndef _WIN32
    #include <unistd.h>
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

const OrtApi* g_ort = NULL;

const OrtApi* gliclass_initialize_ort_api() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return g_ort;
}


bool gliclass_create_inference_config(
    size_t batch_size,
    size_t max_length,
    float threshold,
    char* classification_type,
    bool add_prefix_space,
    GLiClassInferenceConfig* config
) {
    char* template = "Inference parameter is invalid: %s";
    if (batch_size == 0) {
        fprintf(stderr, template, "batch_size shouldn't equal zero");
        return false;
    } else if (max_length == 0) {
        fprintf(stderr, template, "max_length shouldn't equal zero");
        return false;
    } else if (threshold < 0 && threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (threshold < 0 && threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (!(
        strcmp(classification_type, "multi-label") 
        || strcmp(classification_type, "single-label")
    )) {
        fprintf(stderr, template, "classification_type should be equal to 'multi-label' or 'single-label'");
        return false;
    }
    config->batch_size = batch_size;
    config->max_length = max_length;
    config->threshold = threshold;
    config->classification_type = classification_type;
    config->add_prefix_space = add_prefix_space;
    return true;
}


GLiClassSession* gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path,
    const size_t num_threads,
    const bool use_mutex
) {
    // Initializes the ONNX Runtime API
    if (!gliclass_initialize_ort_api()) return false;

    if (num_threads == 0) {
        fprintf(stderr, "num_threads shouldn't equal zero");
        return false;
    }

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) return NULL;

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
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize ONNX environment
    session->env = gliclass_create_ort_env("GLiClass");
    if (!session->env) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize tokenizer
    session->tokenizer = create_tokenizer(tokenizer_path);
    if (!session->tokenizer) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize ONNX session (model loading)
    session->session = gliclass_create_ort_session_cpu_default(
        session->env, model_path, num_threads
    );
    if (!session->session) {
        gliclass_cleanup(session);
        return NULL;
    }

    return session;
}

OrtEnv* gliclass_create_ort_env(const char* env_name) {
    OrtEnv* env = NULL;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, env_name, &env);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create ORT Env: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return NULL;
    }

    return env;
}

#ifdef _WIN32
static wchar_t* convert_path(const char* path) {
    size_t len = mbstowcs(NULL, path, 0);
    if(len == (size_t)-1) {
        fprintf(stderr, "Error: Unable to convert path to wchar_t*: %s\n", path);
        return NULL;
    }

    wchar_t *wide_str = calloc((len + 1), sizeof(wchar_t));
    if(!wide_str) {
        fprintf(stderr, "Error: Unable to convert path to wchar_t*: %s\n", path);
        return NULL;
    }

    mbstowcs(wide_str, path, len + 1);
    return wide_str;
}
#endif


OrtSession* initialize_ort_session(OrtEnv* env, OrtSessionOptions* options, const char* model_path) {
    OrtSession* session = NULL;
    OrtStatus* status = NULL;

    // Load the model and create a session
    #ifdef _WIN32
    wchar_t* path = convert_path(model_path);
    if (!path) {
        g_ort->ReleaseSessionOptions(options);
        return NULL;
    }
    status = g_ort->CreateSession(env, path, options, &session);
    free(path);
    #else
    status = g_ort->CreateSession(env, model_path, options, &session);
    #endif
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create session: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return NULL;
    }
    return session;
}

OrtSessionOptions* initialize_base_options(const int num_threads) {
    OrtSessionOptions* session_options = NULL;
    OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
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
    return session_options;
}


OrtSession* gliclass_create_ort_session_cpu_default(OrtEnv* env, const char* model_path, int num_threads) {
    // Check existence
    if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: Model file not found at path: %s\n", model_path);
        return NULL;
    }

    // Create session options
    OrtSessionOptions* options = initialize_base_options(num_threads);
    if (!options) return NULL;

    // Create session
    OrtSession* session = initialize_ort_session(env, options, model_path);
    g_ort->ReleaseSessionOptions(options);
    return session;
}

#ifdef USE_CUDA
OrtSession* gliclass_create_ort_session_cuda(OrtEnv* env, const char* model_path, int num_threads, int device_id) {
    // Check existence
    if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: Model file not found at path: %s\n", model_path);
        return NULL;
    }

    // Create session options
    OrtSessionOptions* options = initialize_base_options(num_threads);
    if (!options) return NULL;

    // Append CUDA session options
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, device_id);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to add CUDA Execution Provider: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return NULL;
    }
    status = g_ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to enable graph optimization: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return NULL;
    }
    OrtSession* session = initialize_ort_session(env, options, model_path);
    g_ort->ReleaseSessionOptions(options);
    return session;
}
#endif


OrtSession* gliclass_create_ort_session_openvino(OrtEnv* env, const char* model_path, int num_threads, const char* device_type) {
    // Check existence
    if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: Model file not found at path: %s\n", model_path);
        return NULL;
    }

    // Create session options
    OrtSessionOptions* options = initialize_base_options(num_threads);
    if (!options) return NULL;

    // Append OpenVINO EP
    const char* keys[] = {"device_type"};
    const char* values[] = {device_type};
    OrtStatus* status = g_ort->SessionOptionsAppendExecutionProvider_OpenVINO_V2(
        options,
        keys,
        values,
        1 // Number of key-value pairs
    );
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to append OpenVINO EP: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    OrtSession* session = initialize_ort_session(env, options, model_path);
    g_ort->ReleaseSessionOptions(options); // Free session options after creating session
    return session;
}


GLiClassSession* gliclass_init_custom_ort(
    const char* model_config_path,
    const char* tokenizer_path, 
    const bool use_mutex,
    OrtSession* ort_session
) {
    if (!g_ort || !ort_session) {
        fprintf(stderr, "ORT API and ORT session should be initialized!");
        return NULL;
    }

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) return NULL;

    // Initialize ONNX session (model loading)
    session->session = ort_session;

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
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize tokenizer
    session->tokenizer = create_tokenizer(tokenizer_path);
    if (!session->tokenizer) {
        gliclass_cleanup(session);
        return NULL;
    }
    return session;
}


bool gliclass_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
) {
    if (!session || !input_text || !labels || num_labels == 0) {
        fprintf(stderr, "Inputs have invalid value!");
        return false;
    }

    // Allocate output array for results
    *out_num_results = 0;
    *out_results = (GLiClassResult*)calloc(num_labels, sizeof(GLiClassResult));
    if (!*out_results) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }

    char* input = prepare_input(input_text, labels, num_labels, session->model_config->prompt_first, config->add_prefix_space);
    if (!input) {
        fprintf(stderr, "Error while preparing text");
        return false;
    }

    TokenizedInput tokenized = tokenize_input(
        session->tokenizer, 
        (const char*)input, 
        config->max_length
    );

    OrtValue* input_ids_tensor = NULL;
    OrtValue* attention_mask_tensor = NULL;
    prepare_input_tensor(
        &tokenized,
        &input_ids_tensor, 
        &attention_mask_tensor
    );

    OrtValue* output_tensor;
    if (session->use_mutex) {
        #ifndef _WIN32
        pthread_mutex_lock(&queue_mutex);
        #else
        WaitForSingleObject(queue_mutex, INFINITE); 
        #endif
        output_tensor = run_inference(session->session, input_ids_tensor, attention_mask_tensor);
        #ifndef _WIN32
        pthread_mutex_unlock(&queue_mutex);
        #else
        ReleaseMutex(queue_mutex);
        #endif
    } else {
        output_tensor = run_inference(session->session, input_ids_tensor, attention_mask_tensor);
    }
    g_ort->ReleaseValue(input_ids_tensor);
    g_ort->ReleaseValue(attention_mask_tensor);
    free_tokenized_input(&tokenized);
    free(input);

    process_output_tensor(
        session,
        config,
        output_tensor, 
        labels, 
        num_labels, 
        *out_results,
        out_num_results
    );
    return true;
}


bool gliclass_infer_batch(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* input_texts[],
    const size_t num_texts,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult** out_results[],
    size_t* out_num_results[],
    size_t* out_num_results_size
) {
    if (
        !session || !input_texts || !labels || !num_labels || num_labels_size == 0 
        || (num_labels_size != 1 && num_labels_size != num_texts)
    ) {
        fprintf(stderr, "Inputs have invalid value!");
        return false;
    }

    *out_num_results_size = num_texts;
    *out_results = (GLiClassResult**)calloc(*out_num_results_size, sizeof(GLiClassResult*));
    *out_num_results = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!out_num_results || !out_results) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }

    // Allocate memory for tensors
    size_t num_batches = (
        num_texts + config->batch_size - 1
    ) / config->batch_size;
    OrtValue** input_ids_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** attention_mask_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** output_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));

    // Preprocessing stage
    parallel_preprocess(
        session,
        config,
        num_batches,
        input_texts, 
        num_texts,
        labels, 
        num_labels, 
        num_labels_size, 
        input_ids_tensors,
        attention_mask_tensors
    );

    // Inference stage
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        if (session->use_mutex) {
            #ifndef _WIN32
            pthread_mutex_lock(&queue_mutex);
            #else
            WaitForSingleObject(queue_mutex, INFINITE); 
            #endif
            output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
            #ifndef _WIN32
            pthread_mutex_unlock(&queue_mutex);
            #else
            ReleaseMutex(queue_mutex);
            #endif
        } else {
            output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
        }
        g_ort->ReleaseValue(input_ids_tensors[i]);
        g_ort->ReleaseValue(attention_mask_tensors[i]);
    }
    free(input_ids_tensors);
    free(attention_mask_tensors);

    // Postprocess stage - processing batches
    parallel_postprocess(
        session,
        config,
        output_tensors, 
        num_batches,
        num_texts,
        labels, 
        num_labels,
        num_labels_size, 
        config->classification_type,
        *out_results,
        *out_num_results
    );


    return true;
}


void gliclass_free_results(GLiClassResult* results, size_t num_results) {
    if (!results) return;
    free(results);
}


void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size) {
    if (!results) return;
    for (size_t i = 0; i < num_results_size; i++) {
        free(results[i]);
    }
    free(results);
    free(num_results);
}


void gliclass_cleanup(GLiClassSession* session) {
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
    if (session->env) g_ort->ReleaseEnv(session->env);
    if (session->session) g_ort->ReleaseSession(session->session);
    free(session);
}


void gliclass_cleanup_custom_ort(GLiClassSession* session) {
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
    free(session);
}