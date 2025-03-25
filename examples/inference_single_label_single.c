#include <stdio.h>
#include "GLiClass/gliclass_api.h"

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";
    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 0, 2048, 0.5, "multi-label", true, &config
    );

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        8,
        false
    );

    const char* text = "ONNX is an open-source format designed to enable the interoperability of AI models.";

    const char* labels[] = {"format","model","tool","necessity"};

    const size_t num_labels = 4;

    // gliclass_infer(session);
    
    GLiClassResult* results = NULL;
    size_t num_results = 0;
    
    bool ok = gliclass_infer(
        session,
        &config,
        text, 
        labels, 
        num_labels, 
        &results,
        &num_results,
        NULL
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
    return 0;
}