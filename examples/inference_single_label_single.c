#include <stdio.h>
#include "GLiClass/gliclass_api.h"

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";
    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 0, 2048, 0.0, "single-label", true, &config
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
        GC_ONNX_OPENVINO,
        GC_CPU,
        &session
    );
    if (status != GC_OK) {
        fprintf(stderr, "Unable to create session: %s", gliclass_last_error_message());
        gliclass_free_error();
        gliclass_cleanup(session);
    }

    const char* text = "ONNX is an open-source format designed to enable the interoperability of AI models.";

    const char* labels[] = {"format","model","tool","necessity"};

    const size_t num_labels = 4;

    // gliclass_infer(session);
    
    GLiClassResult* results = NULL;
    size_t num_results = 0;
    
    status = gliclass_infer(
        session,
        &config,
        text,
        labels,
        num_labels,
        &results,
        &num_results,
        NULL
    );
    if (status != GC_OK) {
        fprintf(stderr, "Error during inference: %s", gliclass_last_error_message());
        gliclass_free_error();
        return 1;
    }
    
    fprintf(stdout, "\nText: %s\n", text);
    for (size_t i = 0; i < num_results; i++) {
        fprintf(stdout, "Label_%zu: %s, score: %f\n", i, results[i].label, results[i].score);
    }
    gliclass_free_results(results, num_results);
    return 0;
}