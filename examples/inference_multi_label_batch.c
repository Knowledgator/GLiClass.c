#include <stdio.h>
#include "GLiClass/gliclass.h"

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        2, 0, 2048, 0.0, "multi-label", true, &config
    );

    // Initialize session (model setup)
    int num_threads= 8;
    GLiClassSession* session = NULL;
    GLiClassStatus status = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        num_threads,
        false,
        GC_ONNX,
        GC_CPU,
        &session
    );
    if (status != GC_OK) {
        fprintf(stderr, "Unable to create session: %s", gliclass_last_error_message());
        gliclass_free_error();
        return 1;
    }

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

    GLiClassResult** results = NULL;
    size_t* results_shape = NULL;
    size_t results_shape_size = 0;
    
    status = gliclass_infer_batch(
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
        NULL
    );
    if (status != GC_OK) {
        fprintf(stderr, "Error during inference: %s", gliclass_last_error_message());
        gliclass_free_error();
        gliclass_cleanup(session);
        return 1;
    }
    
    for (size_t i = 0; i < results_shape_size; i++) {
        fprintf(stdout, "\nText_%ld/%ld: %s\n", i, results_shape_size, texts[i]);
        for (size_t j = 0; j < results_shape[i]; j++) {
            fprintf(stdout, "Label_%ld: %s, score: %f\n", j, results[i][j].label, results[i][j].score);
        }      
    }

    gliclass_free_results_batch(results, results_shape, results_shape_size);
    gliclass_cleanup(session);
    return 0;
}