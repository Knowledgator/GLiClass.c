#include "GLiClass/gliclass_ort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "preprocessor.h"
#include "error.h"
#include "utils.h"
#include "onnx_runtime/model.h"
#include "onnx_runtime/postprocessor.h"

#ifndef _WIN32
    #include <unistd.h>
    #include <mqueue.h>
    #include <pthread.h>
#else
    #include <io.h>

    #define access _access
    #define F_OK 0
#endif

const OrtApi* g_ort = NULL;

const OrtApi* gliclass_ort_initialize_api() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return g_ort;
}


GLiClassStatus gliclass_ort_create_env(const char* env_name, OrtEnv** env) {
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, env_name, env);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Failed to create ORT Env: %s", msg);
        g_ort->ReleaseStatus(status);
        return GC_PROVIDER_ERROR;
    }
    return GC_OK;
}


GLiClassStatus gliclass_ort_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInput* input,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
);


GLiClassStatus gliclass_ort_infer_batch(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInputs* input,
    const size_t batch_id,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult** out_results[],
    size_t* out_num_results[]
);


void gliclass_ort_cleanup(GLiClassORTSession* session);


void gliclass_ort_cleanup_custom(GLiClassORTSession* session);


#ifdef _WIN32
static GLiClassStatus convert_path(const char* path, wchar_t** output) {
    size_t len = strlen(path);
    if(len == (size_t)-1) {
        set_error("Error: Unable to convert path to wchar_t*: %s", path);
        return GC_LOGICAL_ERROR;
    }

    *output = (wchar_t*)calloc((len + 1), sizeof(wchar_t));
    if(!output) {
        set_error("Unable to allocate wchar path");
        return GC_MEMORY_ERROR;
    }

    size_t converted;
    mbstowcs_s(&converted, *output, len + 1, path, len + 1);
    return GC_OK;
}
#endif


GLiClassStatus initialize_ort_session(
    OrtEnv* env, OrtSessionOptions* options, const char* model_path, OrtSession** session
) {
    OrtStatus* status = NULL;

    // Load the model and create a session
    #ifdef _WIN32
    wchar_t* path = NULL;
    GLiClassStatus gc_status = convert_path(model_path, &path);
    if (gc_status != GC_OK) {
        g_ort->ReleaseSessionOptions(options);
        return gc_status;
    }
    
    status = g_ort->CreateSession(env, path, options, session);
    free(path);
    #else
    status = g_ort->CreateSession(env, model_path, options, session);
    #endif
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to create session: %s", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return GC_PROVIDER_ERROR;
    }
    return GC_OK;
}


GLiClassStatus initialize_base_options(
    const int num_threads, OrtSessionOptions** session_options
) {
    OrtStatus* status = g_ort->CreateSessionOptions(session_options);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to create session options: %s", msg);
        g_ort->ReleaseStatus(status);
        return GC_PROVIDER_ERROR;
    }

    // Set the number of threads for intra-op operations
    status = g_ort->SetIntraOpNumThreads(*session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to set intra-op threads: %s", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(*session_options);
        return GC_PROVIDER_ERROR;
    }

    // Set the number of threads for inter-op operations
    status = g_ort->SetInterOpNumThreads(*session_options, num_threads);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to set inter-op threads: %s", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(*session_options);
        return GC_PROVIDER_ERROR;
    }
    return GC_OK;
}


GLiClassStatus gliclass_ort_cpu_init(
    const char* model_path, int num_threads, GLiClassProviderAPI** provider
) {    
    // Check existence
    if (access(model_path, F_OK) != 0) {
        set_error("Error: Model file not found at path: %s", model_path);
        return GC_FILE_ERROR;
    }
    
    // Create env
    OrtEnv* ort_env = NULL;
    GLiClassStatus status = gliclass_ort_create_env("GLiClass", &ort_env);
    if (status != GC_OK) {
        return status;
    }

    // Create session options
    OrtSessionOptions* options = NULL;
    status = initialize_base_options(num_threads, &options);
    if (status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        return status;
    } 

    // Create session
    OrtSession* ort_session = NULL;
    status = initialize_ort_session(ort_env, options, model_path, &ort_session);
    g_ort->ReleaseSessionOptions(options);
    if (status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        return status;
    }

    // Create provider session
    *provider = (GLiClassProviderAPI*)calloc(1, sizeof(GLiClassProviderAPI*));
    if (!(*provider)) {
        set_error("Unable to allocate provider");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        return GC_MEMORY_ERROR;
    }

    GLiClassORTSession* session = (GLiClassORTSession*)calloc(1, sizeof(GLiClassORTSession));
    if (!session) {
        set_error("Unable to allocate provider session");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        free(*provider);
        return GC_MEMORY_ERROR;
    }
    session->session = ort_session;
    session->env = ort_env;
    
    (*provider)->session = (void*)session;
    (*provider)->run_inference = gliclass_ort_infer;
    (*provider)->run_inference_batch = gliclass_ort_infer_batch;
    (*provider)->cleanup = gliclass_ort_cleanup;
    return GC_OK;
}


#ifndef USE_CUDA
GLiClassStatus gliclass_ort_cuda_init(
    const char* model_path, int num_threads, int device_id, GLiClassProviderAPI** provider
) {
    // Check existence
    if (access(model_path, F_OK) != 0) {
        set_error("Error: Model file not found at path: %s", model_path);
        return GC_FILE_ERROR;
    }

    // Create env
    OrtEnv* ort_env = NULL;
    GLiClassStatus gc_status = gliclass_ort_create_env("GLiClass", &ort_env);
    if (gc_status != GC_OK) {
        return gc_status;
    }

    // Create session options
    OrtSessionOptions* options = NULL;
    gc_status = initialize_base_options(num_threads, &options);
    if (gc_status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        return gc_status;
    }

    // Append CUDA session options
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, device_id);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to add CUDA Execution Provider: %s", msg);
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return GC_PROVIDER_ERROR;
    }
    status = g_ort->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        set_error("Error: Failed to enable graph optimization: %s", msg);
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return GC_PROVIDER_ERROR;
    }

    // Create session
    OrtSession* ort_session = NULL;
    gc_status = initialize_ort_session(ort_env, options, model_path, &ort_session);
    g_ort->ReleaseSessionOptions(options);
    if (gc_status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        return gc_status;
    }

    // Create provider session
    *provider = (GLiClassProviderAPI*)calloc(1, sizeof(GLiClassProviderAPI*));
    if (!(*provider)) {
        set_error("Unable to allocate provider");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        return GC_MEMORY_ERROR;
    }

    GLiClassORTSession* session = (GLiClassORTSession*)calloc(1, sizeof(GLiClassORTSession));
    if (!session) {
        set_error("Unable to allocate provider session");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        free(*provider);
        return GC_MEMORY_ERROR;
    }
    session->session = ort_session;
    session->env = ort_env;
    
    (*provider)->session = (void*)session;
    (*provider)->run_inference = gliclass_ort_infer;
    (*provider)->run_inference_batch = gliclass_ort_infer_batch;
    (*provider)->cleanup = gliclass_ort_cleanup;
    return GC_OK;
}
#endif


GLiClassStatus gliclass_ort_openvino_init(
    const char* model_path, const int num_threads, const char* device_type, 
    GLiClassProviderAPI** provider
) {
    // Check existence
    if (access(model_path, F_OK) != 0) {
        set_error("Error: Model file not found at path: %s", model_path);
        return GC_FILE_ERROR;
    }

    // Create env
    OrtEnv* ort_env = NULL;
    GLiClassStatus gc_status = gliclass_ort_create_env("GLiClass", &ort_env);
    if (gc_status != GC_OK) {
        return gc_status;
    }

    // Create session options
    OrtSessionOptions* options = NULL;
    gc_status = initialize_base_options(num_threads, &options);
    if (gc_status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        return gc_status;
    }

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
        set_error("Failed to append OpenVINO EP: %s", msg);
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(options);
        return GC_PROVIDER_ERROR;
    }

    // Create session
    OrtSession* ort_session = NULL;
    gc_status = initialize_ort_session(ort_env, options, model_path, &ort_session);
    g_ort->ReleaseSessionOptions(options);
    if (gc_status != GC_OK) {
        g_ort->ReleaseEnv(ort_env);
        return gc_status;
    }

    // Create provider session
    *provider = (GLiClassProviderAPI*)calloc(1, sizeof(GLiClassProviderAPI*));
    if (!(*provider)) {
        set_error("Unable to allocate provider");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        return GC_MEMORY_ERROR;
    }

    GLiClassORTSession* session = (GLiClassORTSession*)calloc(1, sizeof(GLiClassORTSession));
    if (!session) {
        set_error("Unable to allocate provider session");
        g_ort->ReleaseEnv(ort_env);
        g_ort->ReleaseSession(ort_session);
        free(*provider);
        return GC_MEMORY_ERROR;
    }
    session->session = ort_session;
    session->env = ort_env;
    
    (*provider)->session = (void*)session;
    (*provider)->run_inference = gliclass_ort_infer;
    (*provider)->run_inference_batch = gliclass_ort_infer_batch;
    (*provider)->cleanup = gliclass_ort_cleanup;
    return GC_OK;
}


GLiClassStatus gliclass_ort_init_custom(
    OrtSession* ort_session, GLiClassProviderAPI** provider
) {
    if (!g_ort || !ort_session) {
        set_error("ORT API and ORT session should be initialized!");
        return GC_LOGICAL_ERROR;
    }

    *provider = (GLiClassProviderAPI*)calloc(1, sizeof(GLiClassProviderAPI));
    if (!(*provider)) {
        set_error("Unable to allocate provider");
        return GC_MEMORY_ERROR;
    }

    GLiClassORTSession* session = (GLiClassORTSession*)calloc(1, sizeof(GLiClassORTSession));
    if (!session) {
        set_error("Unable to allocate provider session");
        free(*provider);
        return GC_MEMORY_ERROR;
    }
    session->session = (void*)ort_session;
    session->env = NULL;

    (*provider)->session = (void*)session;
    (*provider)->run_inference = gliclass_ort_infer;
    (*provider)->run_inference_batch = gliclass_ort_infer_batch;
    (*provider)->cleanup = gliclass_ort_cleanup_custom;
    return GC_OK;
}


GLiClassStatus gliclass_ort_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInput* input,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
) {
    OrtValue* input_ids_tensor = NULL;
    OrtValue* attention_mask_tensor = NULL;

    GLiClassStatus status = ort_prepare_input_tensors(
        input,
        &input_ids_tensor, 
        &attention_mask_tensor
    );
    if (status != GC_OK) return status;

    OrtValue* output_tensor = NULL;
    if (session->use_mutex) {
        lock_mutex();
        status = ort_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
        unlock_mutex();
    } else {
        status = ort_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
    }
    g_ort->ReleaseValue(input_ids_tensor);
    g_ort->ReleaseValue(attention_mask_tensor);

    if (status != GC_OK) {
        return status;
    }

    return ort_process_output_tensor(
        config,
        output_tensor, 
        labels, 
        num_labels, 
        *out_results,
        out_num_results
    );
}


GLiClassStatus gliclass_ort_infer_batch(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInputs* input,
    const size_t batch_id,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult** out_results[],
    size_t* out_num_results[]
) {
    OrtValue* input_ids_tensor = NULL;
    OrtValue* attention_mask_tensor = NULL;
    GLiClassStatus status = ort_prepare_input_tensors_batch(
        input,
        &input_ids_tensor, 
        &attention_mask_tensor
    );
    if (status != GC_OK) return status;

    OrtValue* output_tensor = NULL;
    if (session->use_mutex) {
        lock_mutex();
        status = ort_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
        unlock_mutex();
    } else {
        status = ort_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
    }
    g_ort->ReleaseValue(input_ids_tensor);
    g_ort->ReleaseValue(attention_mask_tensor);
    if (status != GC_OK) return status;

    status = ort_process_output_tensor_batch(
        config,
        output_tensor, 
        labels, 
        num_labels, 
        num_labels_size, 
        batch_id, 
        *out_results,
        *out_num_results
    );
    // Free output tensor after processing
    g_ort->ReleaseValue(output_tensor);
    return status;
}


void gliclass_ort_cleanup(GLiClassORTSession* session) {
    if (!session) return;
    if (session->env) g_ort->ReleaseEnv(session->env);
    if (session->session) g_ort->ReleaseSession(session->session);
    free(session);
}


void gliclass_ort_cleanup_custom(GLiClassORTSession* session) {
    free(session);
}