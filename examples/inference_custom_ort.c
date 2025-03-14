#include <stdio.h>
#include "onnxruntime_c_api.h"
#include "gliclass_api.h"
#ifndef _WIN32
    #include <unistd.h>
#else
    #include <io.h>

    #define access _access
    #define F_OK 0
#endif


#ifdef _WIN32
wchar_t* convert_path(const char* path) {
    size_t len = mbstowcs(NULL, path, 0);
    if(len == (size_t)-1) {
        fprintf(stderr, "Error: Unable to convert path to wchar_t*: %s\n", path);
        return NULL;
    }

    wchar_t *wide_str = malloc((len + 1) * sizeof(wchar_t));
    if(!wide_str) {
        fprintf(stderr, "Error: Unable to convert path to wchar_t*: %s\n", path);
        return NULL;
    }

    mbstowcs(wide_str, path, len + 1);
    return wide_str;
}
#endif

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    size_t num_threads = 8;

    InferenceConfig config = {
        8, 2048, 0.5, "multi-label", false
    };

    // Initializes the ONNX Runtime API
    if (!initialize_ort_api()) return false;

    OrtEnv* ort_env = NULL;
    OrtSessionOptions* ort_session_options = NULL;
    OrtSession* ort_session = NULL;
    OrtStatus* ort_status = NULL;


    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "GLiClass", &ort_env);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: Failed to create env for ONNX Runtime: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return 1;
    } 

    // Check existence
    if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: Model file not found at path: %s\n", model_path);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }

    // Create session options
    ort_status = g_ort->CreateSessionOptions(&ort_session_options);
    if (ort_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(ort_status);
        fprintf(stderr, "Error: Failed to create session options: %s\n", msg);
        g_ort->ReleaseStatus(ort_status);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }

    // Set the number of threads for intra-op operations
    ort_status = g_ort->SetIntraOpNumThreads(ort_session_options, num_threads);
    if (ort_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(ort_status);
        fprintf(stderr, "Error: Failed to set intra-op threads: %s\n", msg);
        g_ort->ReleaseStatus(ort_status);
        g_ort->ReleaseSessionOptions(ort_session_options);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }

    // Set the number of threads for inter-op operations
    ort_status = g_ort->SetInterOpNumThreads(ort_session_options, num_threads);
    if (ort_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(ort_status);
        fprintf(stderr, "Error: Failed to set inter-op threads: %s\n", msg);
        g_ort->ReleaseStatus(ort_status);
        g_ort->ReleaseSessionOptions(ort_session_options);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }

    #ifdef USE_CUDA // GPU
    int device_id = 0;
    ort_status = OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, device_id);
    if (ort_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(ort_status);
        fprintf(stderr, "Error: Failed to add CUDA Execution Provider: %s\n", msg);
        g_ort->ReleaseStatus(ort_status);
        g_ort->ReleaseSessionOptions(ort_session_options);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }
    g_ort->SetSessionGraphOptimizationLevel(ort_session_options, ORT_ENABLE_ALL);
    printf("\tCUDA Execution Provider added successfully.\n");
    #else
    printf("\tUsing CPU Execution Provider.\n");
    #endif

    // Load the model and create a session
    #ifdef _WIN32
    wchar_t* path = convert_path(model_path);
    if (!path) {
        g_ort->ReleaseSessionOptions(ort_session_options);
        g_ort->ReleaseEnv(ort_env);
        return 1;
    }
    ort_status = g_ort->CreateSession(env, path, session_options, &session);
    free(path);
    #else
    ort_status = g_ort->CreateSession(ort_env, model_path, ort_session_options, &ort_session);
    #endif
    if (ort_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(ort_status);
        fprintf(stderr, "Error: Failed to create session: %s\n", msg);
        g_ort->ReleaseStatus(ort_status);
        g_ort->ReleaseSessionOptions(ort_session_options);
        return 1;
    }
    g_ort->ReleaseSessionOptions(ort_session_options);

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init_custom_ort(
        model_config_path,
        tokenizer_path,
        &config,
        ort_session
    );

    const char* text = "ONNX is an open-source format designed to enable the interoperability of AI models.";
    const char* labels[] = {"format","model","tool","necessity"};
    const size_t num_labels = 4;

    GLiClassResult* results = NULL;
    size_t num_results = 0;
    
    bool ok = gliclass_infer(
        session, 
        text, 
        labels, 
        num_labels, 
        &results,
        &num_results
    );

    if (!ok) {
        fprintf(stderr, "Errors occur during inference!");
        gliclass_cleanup(session);
        return 1;
    }
    
    fprintf(stdout, "\nText: %s\n", text);
    for (size_t i = 0; i < num_results; i++) {
        fprintf(stdout, "Label_%ld: %s, score: %f\n", i, results[i].label, results[i].score);
    }
    gliclass_free_results(results, num_results);
    gliclass_cleanup(session);

    return 0;
}