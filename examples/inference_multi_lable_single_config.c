#include <stdio.h>
#include "gliclass_api.h"

// Function to run a single inference with a given config
static void run_inference_with_config(InferenceConfig config, const char* model_path, const char* model_config_path, const char* tokenizer_path, const char* text, const char* labels[], size_t num_labels) {
    printf("\nRunning inference with threshold: %.3f\n", config.threshold);

    // Initialize session
    GLiClassSession* session = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        &config,
        8 // Number of threads
    );

    if (!session) {
        fprintf(stderr, "Failed to initialize GLiClass session\n");
        return;
    }

    // Run inference
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
        fprintf(stderr, "Errors occurred during inference!\n");
        gliclass_cleanup(session);
        return;
    }

    // Output results
    printf("\nText: %s\n", text);
    for (size_t i = 0; i < num_results; i++) {
        printf("Label_%ld: %s, score: %.4f\n", i, results[i].label, results[i].score);
    }

    // Cleanup
    gliclass_free_results(results, num_results);
    gliclass_cleanup(session);
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

    // Array of different InferenceConfig settings to test
    InferenceConfig configs[] = {
        {8, 2048, 0.5, "multi-label", false},
        {8, 2048, 0.1, "multi-label", false},
        {8, 2048, 0.01, "multi-label", false}
    };

    const size_t num_configs = sizeof(configs) / sizeof(configs[0]);

    // Loop through each config and run inference
    for (size_t i = 0; i < num_configs; ++i) {
        run_inference_with_config(configs[i], model_path, model_config_path, tokenizer_path, text, labels, num_labels);
    }

    return 0;
}
