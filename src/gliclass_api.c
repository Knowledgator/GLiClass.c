#include "GLiClass/gliclass_api.h"
#include "read_data.h"
#include "error.h"

#ifndef USE_ONNX
#include "GLiClass/gliclass_ort.h"
#endif

#ifndef USE_OPENVINO
#include "GLiClass/gliclass_ov.h"
#endif

// Mutex declarations
#ifndef _WIN32
static pthread_mutex_t queue_mutex;
#else
static HANDLE queue_mutex;
#endif

GLiClassStatus initialize_model_config(const char* model_config_path, GLiClassModelConfig** model_config_out) {
    char** json_string = NULL;
    GLiClassStatus status = read_file(model_config_path, json_string);
    if (status != OK) return status;

    status = parse_model_config_json(json_string, model_config_out);
    free((void*)json_string);
    return status;
}

GLiClassStatus gliclass_init_custom_provider(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const bool use_mutex,
    void* provider_session,
    GLiClassSession** session_out
) {
    if (num_threads == 0) {
        set_error("num_threads shouldn't equal zero");
        return LOGICAL_ERROR;
    }

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) {
        fprintf(stderr, "Unable to allocate session");
        return MEMORY_ERROR;
    };

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
    GLiClassStatus status = initialize_model_config(model_config_path, &(session->model_config));
    if (status != OK) {
        gliclass_cleanup(session);
        return status;
    }

    // Initialize tokenizer
    status = create_tokenizer(tokenizer_path, &(session->tokenizer));
    if (status != OK) {
        gliclass_cleanup(session);
        return status;
    }

    if (!provider_session) {
        set_error("No provider was provided");
        return LOGICAL_ERROR;
    }
    session->provider_session = provider_session;
    *session_out = session;
    return status;
}

GLiClassStatus gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const bool use_mutex,
    GLiClassProvider provider,
    GLiClassDevice device,
    GLiClassSession** session_out
) {
    void* provider_session = NULL;
    switch (provider) {
    case ONNX:
        #ifndef USE_ONNX

        if (device == GPU) {
            #ifndef USE_CUDA

            #else
            
            #endif
        }
        
        #else
        set_error("ONNX provider is not supported for current build");
        return PROVIDER_ERROR;
        #endif
    }

    if (!provider_session) {
        set_error("No provider was provided");
        return LOGICAL_ERROR;
    }
    session->provider_session = provider_session;
    *session_out = session;
    return status;
}