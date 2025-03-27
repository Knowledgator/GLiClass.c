#include "GLiClass/gliclass_ov.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "preprocessor.h"
#include "utils.h"
#include "error.h"
#include "openvino_runtime/model.h"
#include "openvino_runtime/postprocessor.h"

#ifndef _WIN32
    #include <unistd.h>
    #include <mqueue.h>
    #include <pthread.h>
#else
    #include <io.h>

    #define access _access
    #define F_OK 0
#endif

GLiClassStatus gliclass_openvino_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInput* input,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
);


void gliclass_openvino_cleanup(GLiClassOpenVinoSession* session);


GLiClassStatus gliclass_openvino_init(
    const char* model_path,
    const int num_threads,
    const char* device_type,
    GLiClassProviderAPI** provider
) {
    GLiClassOpenVinoSession* session = (GLiClassOpenVinoSession*)calloc(1, sizeof(GLiClassOpenVinoSession));
    if (!session) {
        set_error("Unable to allocate session");
        return GC_MEMORY_ERROR;
    }

    session->core = NULL;
    ov_status_e status = ov_core_create(&(session->core));
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        set_error("Unable to init OpenVino core: %s: %s", e, ov_get_last_err_msg());
        gliclass_openvino_cleanup(session);
        return GC_PROVIDER_ERROR;
    }

    char threads[3];
    snprintf(threads, 3, "%d", num_threads);
    session->model = NULL;
    if (num_threads > 0 && strcmp(device_type, "CPU") == 0) {
        status = ov_core_compile_model_from_file(
            session->core, model_path, device_type, 4, &(session->model),
            ov_property_key_hint_performance_mode, "LATENCY",
            ov_property_key_inference_num_threads, threads
        );
    } else {
        status = ov_core_compile_model_from_file(
            session->core, model_path, device_type, 2, &session->model,
            ov_property_key_hint_performance_mode, "LATENCY"
        );
    }

    if (status != OK) {
        const char* e = ov_get_error_info(status);
        set_error("Unable to compile model: %s: %s\n", e, ov_get_last_err_msg());
        gliclass_openvino_cleanup(session);
        return GC_PROVIDER_ERROR;
    }

    *provider = (GLiClassProviderAPI*)calloc(1, sizeof(GLiClassProviderAPI));
    if (!provider) {
        set_error("Unable to allocate provider");
        return GC_MEMORY_ERROR;
    }

    (*provider)->session = (void*)session;
    (*provider)->run_inference = gliclass_openvino_infer;
    // (*provider)->run_inference_batch = gliclass_openvino_infer_batch;
    (*provider)->run_inference_batch = NULL;
    (*provider)->cleanup = gliclass_openvino_cleanup;
    return OK;
}


GLiClassStatus gliclass_openvino_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const TokenizedInput* input,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
) {
    ov_tensor_t* input_ids_tensor = NULL;
    ov_tensor_t* attention_mask_tensor = NULL;
    GLiClassStatus status = openvino_prepare_input_tensors(
        input,
        &input_ids_tensor, 
        &attention_mask_tensor
    );
    if (status != OK) return status;

    ov_tensor_t* output_tensor = NULL;
    if (session->use_mutex) {
        lock_mutex();
        status = openvino_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
        unlock_mutex();
    } else {
        status = openvino_run_inference(
            session->provider->session, input_ids_tensor, attention_mask_tensor, &output_tensor
        );
    }
    ov_tensor_free(input_ids_tensor);
    ov_tensor_free(attention_mask_tensor);
    if (status != OK) return status;
    
    status = openvino_process_output_tensor(
        session,
        config,
        output_tensor, 
        labels, 
        num_labels, 
        *out_results,
        out_num_results
    );
    ov_tensor_free(output_tensor);
    return status;
}


void gliclass_openvino_cleanup(GLiClassOpenVinoSession* session) {
    if (!session) return;
    if (session->model) ov_compiled_model_free(session->model);
    if (session->core) ov_core_free(session->core);
    free(session);
}