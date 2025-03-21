#include <stdio.h>
#include "onnxruntime_c_api.h"
#include "GLiClass/gliclass_api.h"
#include <openvino/c/openvino.h>
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

    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    int num_threads = 8;

    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 2048, 0.5, "multi-label", true, &config
    );

    // Initializes the ONNX Runtime API
    if (!gliclass_initialize_ort_api()) return false;

    OrtEnv* ort_env = gliclass_create_ort_env("GLiClass");
    OrtSession* ort_session = gliclass_create_ort_session_openvino(ort_env, model_path, num_threads, "GPU");

    if (!ort_env || !ort_session) {
        fprintf(stderr, "ERROR WITH ORT!");
        return 1;
    }

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init_custom_ort(
        model_config_path,
        tokenizer_path,
        false, // use mutex
        ort_session
    );

    const char* text = "ONNX is an open-source format designed to enable the interoperability of AI models.";
    const char* labels[] = {"format","model","tool","necessity"};
    const size_t num_labels = 4;

    GLiClassResult* results = NULL;
    size_t num_results = 0;
    bool truncated = false;
    
    double time = (double)clock() / CLOCKS_PER_SEC;
    bool ok = gliclass_infer(
        session,
        &config,
        text, 
        labels, 
        num_labels, 
        &results,
        &num_results,
        &truncated
    );
    time = (double)clock() / CLOCKS_PER_SEC - time;
    fprintf(stdout, "Elapsed: %f s\n", time);

    if (!ok) {
        fprintf(stderr, "Errors occured during inference!");
        gliclass_cleanup(session);
        return 1;
    }
    
    fprintf(stdout, "\nText: %s\n", text);
    fprintf(stdout, "\nTruncated: %s\n", truncated ? "true": "false");
    for (size_t i = 0; i < num_results; i++) {
        fprintf(stdout, "Label_%ld: %s, score: %f\n", i, results[i].label, results[i].score);
    }
    gliclass_free_results(results, num_results);
    gliclass_cleanup(session);

    return 0;
}