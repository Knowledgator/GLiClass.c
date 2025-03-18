#include "gliclass_api.h"

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

bool initialize_ort_api() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return (bool)g_ort;
}

bool validate_inference_config(const InferenceConfig* inference_config) {
    char* template = "Inference parameter is invalid: %s";
    if (inference_config->batch_size == 0) {
        fprintf(stderr, template, "batch_size shouldn't equal zero");
        return false;
    } else if (inference_config->max_length == 0) {
        fprintf(stderr, template, "max_length shouldn't equal zero");
        return false;
    } else if (inference_config->threshold < 0 && inference_config->threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (inference_config->threshold < 0 && inference_config->threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (!(
        strcmp(inference_config->classification_type, "multi-label") 
        || strcmp(inference_config->classification_type, "single-label")
    )) {
        fprintf(stderr, template, "classification_type should be equal to 'multi-label' or 'single-label'");
        return false;
    }
    return true;
}

// TODO: add overload with inferance config + validate it + parse model config 
GLiClassSession* gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path, 
    const InferenceConfig* inference_config,
    const size_t num_threads
) {
    // Initializes the ONNX Runtime API
    if (!initialize_ort_api()) return false;

    if (num_threads == 0) {
        fprintf(stderr, "num_threads shouldn't equal zero");
        return false;
    }

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) return NULL;

    if (!validate_inference_config(inference_config)) {
        return NULL;
    }
    session->inference_config = inference_config;

    // Initialize the model config
    session->model_config = initialize_model_config(model_config_path);
    if (!session->model_config) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize ONNX environment
    session->env = initialize_ort_environment();
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
    session->session = create_ort_session(
        session->env, model_path, num_threads
    );
    if (!session->session) {
        gliclass_cleanup(session);
        return NULL;
    }

    return session;
}

OrtEnv* create_ort_env(const char* env_name) {
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
wchar_t* convert_path(const char* path) {
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

OrtSession* create_ort_session_with_openvino(OrtEnv* env, const char* model_path, int num_threads, const char* device_type) {
     // Check existence
     if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: Model file not found at path: %s\n", model_path);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    OrtSessionOptions* session_options = NULL;
    OrtStatus* status = g_ort->CreateSessionOptions(&session_options);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create ORT SessionOptions: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    // Set the number of threads for intra-op operations
    status = g_ort->SetIntraOpNumThreads(session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to set intra-op threads: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    // Set the number of threads for inter-op operations
    status = g_ort->SetInterOpNumThreads(session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to set inter-op threads: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    // Append OpenVINO EP
    const char* keys[] = {"device_type"};
    const char* values[] = {device_type};

    status = g_ort->SessionOptionsAppendExecutionProvider_OpenVINO_V2(
        session_options,
        keys,
        values,
        1 // Number of key-value pairs
    );
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to append OpenVINO EP: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    OrtSession* session = NULL;    
    // Load the model and create a session
    #ifdef _WIN32
    wchar_t* path = convert_path(model_path);
    if (!path) {
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }

    // Create session
    status = g_ort->CreateSession(env, path, session_options, &session);
    free(path);
    #else
    status = g_ort->CreateSession(env, model_path, session_options, &session);
    #endif
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create ORT Session: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return NULL;
    }
    g_ort->ReleaseSessionOptions(session_options); // Free session options after creating session

    return session;
}

GLiClassSession* gliclass_init_custom_ort(
    const char* model_config_path,
    const char* tokenizer_path, 
    const InferenceConfig* inference_config,
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

    if (!validate_inference_config(inference_config)) {
        return NULL;
    }
    session->inference_config = inference_config;

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

// single process
bool gliclass_infer(
    GLiClassSession* session,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
) {
    if (!session || !input_text || !labels || num_labels == 0) return false;

    // Initialize queue mutex
    #ifndef _WIN32
    pthread_mutex_init(&queue_mutex, NULL);
    #else
    queue_mutex = CreateMutex(NULL, FALSE, NULL);
    #endif

    // Allocate output array for results
    *out_num_results = 0;
    *out_results = (GLiClassResult*)calloc(num_labels, sizeof(GLiClassResult));
    if (!*out_results) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }

    char* input = prepare_input(input_text, labels, num_labels, session->model_config->prompt_first, session->inference_config->add_prefix_space);
    if (!input) {
        fprintf(stderr, "Error while preparing text");
        return false;
    }

    TokenizedInput tokenized = tokenize_input(
        session->tokenizer, 
        (const char*)input, 
        session->inference_config->max_length
    );

    OrtValue* input_ids_tensor = NULL;
    OrtValue* attention_mask_tensor = NULL;
    prepare_input_tensor(
        &tokenized,
        &input_ids_tensor, 
        &attention_mask_tensor
    );


    #ifdef USE_CUDA // GPU
    pthread_mutex_lock(&queue_mutex);
    OrtValue* output_tensor = run_inference(session->session, input_ids_tensor, attention_mask_tensor);
    pthread_mutex_unlock(&queue_mutex);
    #else
    OrtValue* output_tensor = run_inference(session->session, input_ids_tensor, attention_mask_tensor);
    #endif
    g_ort->ReleaseValue(input_ids_tensor);
    g_ort->ReleaseValue(attention_mask_tensor);
    free_tokenized_input(&tokenized);
    free(input);

    process_output_tensor(
        session,
        output_tensor, 
        labels, 
        num_labels, 
        *out_results,
        out_num_results
    );

    #ifndef _WIN32
	pthread_mutex_destroy(&queue_mutex);
    #else
	CloseHandle(queue_mutex);
    #endif
    return true;
}

// batch process
bool gliclass_infer_batch(
    GLiClassSession* session,
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
    ) return false;

    // Initialize queue mutex
    #ifndef _WIN32
    pthread_mutex_init(&queue_mutex, NULL);
    #else
    queue_mutex = CreateMutex(NULL, FALSE, NULL);
    #endif

    *out_num_results_size = num_texts;
    *out_results = (GLiClassResult**)calloc(*out_num_results_size, sizeof(GLiClassResult*));
    *out_num_results = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!out_num_results || !out_results) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }

    // Allocate memory for tensors
    size_t num_batches = (
        num_texts + session->inference_config->batch_size - 1
    ) / session->inference_config->batch_size;
    OrtValue** input_ids_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** attention_mask_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** output_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));

    // Preprocessing stage
    parallel_preprocess(
        session,
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
        #ifdef USE_CUDA // GPU
        pthread_mutex_lock(&queue_mutex);
        output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
        pthread_mutex_unlock(&queue_mutex);
        #else
        output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
        #endif
        g_ort->ReleaseValue(input_ids_tensors[i]);
        g_ort->ReleaseValue(attention_mask_tensors[i]);
    }
    free(input_ids_tensors);
    free(attention_mask_tensors);

    // Postprocess stage - processing batches
    parallel_postprocess(
        session,
        output_tensors, 
        num_batches,
        num_texts,
        labels, 
        num_labels,
        num_labels_size, 
        session->inference_config->classification_type,
        *out_results,
        *out_num_results
    );

    #ifndef _WIN32
	pthread_mutex_destroy(&queue_mutex);
    #else
	CloseHandle(queue_mutex);
    #endif
    return true;
}

void gliclass_free_results(GLiClassResult* results, size_t num_results) {
    if (!results) return;
    // for (size_t i = 0; i < num_results; i++) {
    //     free(results[i].label);
    // }
    free(results);
}

void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size) {
    if (!results) return;
    for (size_t i = 0; i < num_results_size; i++) {
        // for (size_t j = 0; j < num_results[i]; j++) {
        //     free(results[i][j].label);
        // }
        free(results[i]);
    }
    free(results);
    free(num_results);
}

void gliclass_cleanup(GLiClassSession* session) {
    if (!session) return;
    if (session->model_config) free((void*)session->model_config);
    if (session->tokenizer) tokenizers_free(session->tokenizer);
    if (session->env) g_ort->ReleaseEnv(session->env);
    if (session->session) g_ort->ReleaseSession(session->session);
    free(session);
}

void gliclass_cleanup_custom_ort(GLiClassSession* session) {
    if (!session) return;
    if (session->model_config) free((void*)session->model_config);
    if (session->tokenizer) tokenizers_free(session->tokenizer);
    free(session);
}