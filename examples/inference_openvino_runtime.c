#include <stdio.h>
#include "GLiClass/gliclass.h"
#include <time.h>

void list_devices() {
    ov_core_t* core = NULL;
    ov_core_create(&core);
    if (!core) {
        fprintf(stderr, "Error: failed to create ov_core\n");
        return;
    }

    // Get the available devices.
    ov_available_devices_t available_devices;
    ov_status_e status = ov_core_get_available_devices(core, &available_devices);
    if (status != OK) {
        fprintf(stderr, "Error: failed to get available devices (status=%d)\n", status);
        ov_core_free(core);
        return;
    }

    // Print each device.
    printf("Available devices:\n");
    for (size_t i = 0; i < available_devices.size; i++) {
        fprintf(stdout, "  %s\n", available_devices.devices[i]);
    }

    // Free allocated resources.
    ov_available_devices_free(&available_devices);
    ov_core_free(core);
}

int main() {
    list_devices();
    ov_version_t version;
    ov_get_openvino_version(&version);
    fprintf(stdout, "%s\n", version.buildNumber);
    ov_version_free(&version);

    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 0, 2048, 0.0, "multi-label", true, &config
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
        GC_OPENVINO,
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

    GLiClassResult* results = NULL;
    GLiClassTokensInfo info;
    size_t num_results = 0;
    
    double time = (double)clock() / CLOCKS_PER_SEC;
    status = gliclass_infer(
        session,
        &config,
        text,
        labels,
        num_labels,
        &results,
        &num_results,
        &info
    );
    time = (double)clock() / CLOCKS_PER_SEC - time;
    fprintf(stdout, "Elapsed: %f s\n", time);

    if (status != GC_OK) {
        fprintf(stderr, "Error during inference: %s", gliclass_last_error_message());
        gliclass_free_error();
        return 1;
    }
    
    fprintf(stdout, "\nText: %s\n", text);
    fprintf(stdout, "\nTruncated: %s\n", info.truncated ? "true": "false");
    fprintf(stdout, "\nProcessed tokens: %zu\n", info.tokens_num);
    for (size_t i = 0; i < num_results; i++) {
        fprintf(stdout, "Label_%zu: %s, score: %f\n", i, results[i].label, results[i].score);
    }
    gliclass_free_results(results, num_results);
    
    gliclass_cleanup(session);
    return 0;
}