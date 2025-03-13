#include <stdio.h>
#include "gliclass_api.h"

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";
    InferenceConfig config = {
        8, 2048, 0.5, "single-label"
    };

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        &config,
        8
    );

    const char* texts[] = {
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Support Ukraine"
    };

    const char* group1[] = {"format","model","tool","necessity"};
    const char* group2[] = {"question","tool","statement"};
    const char* group3[] = {"call to action", "necessity"};
    const char** labels[] = {group1, group2, group3};

    const size_t labels_shape[] = {4, 3, 2};
    const size_t labels_shape_size = 3;

    // gliclass_infer(session);
    
    GLiClassResult** results = NULL;
    size_t* results_shape = NULL;
    size_t results_shape_size = 0;
    
    bool ok = gliclass_infer_batch(
        session, 
        texts, 
        3, 
        labels, 
        labels_shape, 
        labels_shape_size,
        &results,
        &results_shape,
        &results_shape_size // TODO: remove?
    );

    if (!ok) {
        fprintf(stderr, "Errors occur during inference!");
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