#include <stdio.h>
#include "onnxruntime_c_api.h"
#include "GLiClass/gliclass_api.h"
#include <time.h>

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    int num_threads = 8;
    int device_id = 0;
    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 2048, 0.5, "multi-label", true, &config
    );

    // Initializes the ONNX Runtime API
    if (!gliclass_initialize_ort_api()) return false;

    OrtEnv* ort_env = gliclass_create_ort_env("GLiClass");
    OrtSession* ort_session = gliclass_create_ort_session_cuda(ort_env, model_path, num_threads, device_id);

    if (!ort_env || !ort_session) {
        fprintf(stderr, "ERROR WITH ORT!");
        return 1;
    }

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init_custom_ort(
        model_config_path,
        tokenizer_path,
        true, // use mutex lock for inference
        ort_session
    );

    const char* texts[] = {
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Support Ukraine"
    };
    size_t num_texts = 3;

    const char* group1[] = {"format","model","tool","cat"};
    const char* group2[] = {"question","tool","statement"};
    const char* group3[] = {"call to action", "necessity"};
    const char** labels[] = {group1, group2, group3};

    const size_t labels_shape[] = {4, 3, 2};
    const size_t labels_shape_size = 3;

    // gliclass_infer(session);
    
    GLiClassResult** results = NULL;
    size_t* results_shape = NULL;
    size_t results_shape_size = 0;
    bool* truncated = NULL; 
    
    double time = (double)clock() / CLOCKS_PER_SEC;
    bool ok = gliclass_infer_batch(
        session, 
        &config,
        texts, 
        num_texts, 
        labels, 
        labels_shape, 
        labels_shape_size,
        &results,
        &results_shape,
        &results_shape_size,
        &truncated
    );
    time = (double)clock() / CLOCKS_PER_SEC - time;
    fprintf(stdout, "Elapsed: %f s\n", time);

    if (!ok) {
        fprintf(stderr, "Errors occured during inference!");
        gliclass_cleanup(session);
        return 1;
    }
    
    for (size_t i = 0; i < results_shape_size; i++) {
        fprintf(stdout, "\nText_%ld/%ld: %s\n", i, results_shape_size, texts[i]);
        fprintf(stdout, "\nTruncated: %s\n", truncated[i] ? "true": "false");
        for (size_t j = 0; j < results_shape[i]; j++) {
            fprintf(stdout, "Label_%ld: %s, score: %f\n", j, results[i][j].label, results[i][j].score);
        }      
    }
    gliclass_free_results_batch(results, results_shape, results_shape_size);
    gliclass_cleanup(session);
    return 0;
}