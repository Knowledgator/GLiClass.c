#include <stdio.h>
#include "GLiClass/gliclass_api.h"

// Function to run a single inference with a given config
static bool run_inference_with_config(
    GLiClassSession* session,
    GLiClassInferenceConfig* config, 
    const char* text, 
    const char* labels[], 
    size_t num_labels
) {
    printf("\nRunning inference with threshold: %.3f\n", config->threshold);

    // Run inference
    GLiClassResult* results = NULL;
    size_t num_results = 0;

    bool ok = gliclass_infer(
        session,
        config,
        text,
        labels,
        num_labels,
        &results,
        &num_results
    );

    if (!ok) {
        fprintf(stderr, "Errors occurred during inference!\n");
        return ok;
    }

    // Output results
    printf("\nText: %s\n", text);
    for (size_t i = 0; i < num_results; i++) {
        printf("Label_%ld: %s, score: %.4f\n", i, results[i].label, results[i].score);
    }

    // Cleanup
    gliclass_free_results(results, num_results);
    return ok;
}


int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    const char* text = "I'm in a reading rut and need recommendations about books! I typically enjoy fantasy sagas like Lord of the Rings or Game of Thrones, but I want something fresh. Any new releases worth checking out? I'll take anything with a well-built world and gripping plot. Help a fellow book lover out!";

    const char* labels[] = {
        "Automobile",
        "Beauty",
        "Books",
        "Business",
        "Computers",
        "Education",
        "Electronics",
        "Entertainment",
        "Finance",
        "Fitness"
    };

    const size_t num_labels = 10;

    // Initialize session
    GLiClassSession* session = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        8 // Number of threads
    );
    if (!session) {
        fprintf(stderr, "Failed to initialize GLiClass session\n");
        return 1;
    }

    // Array of different InferenceConfig settings to test
    GLiClassInferenceConfig configs[] = {
        {8, 2048, 0.5, "multi-label", false},
        {8, 2048, 0.1, "multi-label", false},
        {8, 2048, 0.01, "multi-label", false}
    };

    const size_t num_configs = sizeof(configs) / sizeof(configs[0]);

    // Loop through each config and run inference
    for (size_t i = 0; i < num_configs; ++i) {
        bool ok = run_inference_with_config(session, &configs[i], text, labels, num_labels);
        if (!ok) {
            gliclass_cleanup(session);
            return 1;
        }
    }
    gliclass_cleanup(session);
    return 0;
}
