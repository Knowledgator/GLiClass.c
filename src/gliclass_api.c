#include "GLiClass/gliclass_api.h"

#include <stdlib.h>

#include "read_data.h"
#include "error.h"
#include "utils.h"
#include "preprocessor.h"
#include "tokenizer.h"

#include "GLiClass/gliclass.h"

GLiClassStatus* initialize_model_config(const char* model_config_path, GLiClassModelConfig** model_config_out) {
    char* json_string = NULL;
    GLiClassStatus* status = read_file(model_config_path, &json_string);
    if (status != NULL) return status;

    status = parse_model_config_json(json_string, model_config_out);
    free((void*)json_string);
    return status;
}

GLiClassStatus* gliclass_init_custom_provider(
    const char* model_config_path,
    const char* tokenizer_path,
    const bool use_mutex,
    GLiClassProviderAPI* provider,
    GLiClassSession** session_out
) {
    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate session");
    };

    session->use_mutex = use_mutex;
    if (session->use_mutex) {
        init_mutex();
    }

    // Initialize the model config
    GLiClassStatus* status = initialize_model_config(model_config_path, &(session->model_config));
    if (status != NULL) {
        gliclass_cleanup(session);
        return status;
    }

    // Initialize tokenizer
    status = create_tokenizer(tokenizer_path, &(session->tokenizer));
    if (status != NULL) {
        gliclass_cleanup(session);
        return status;
    }

    if (!provider) {
        return set_error(GC_LOGICAL_ERROR, "No provider was provided");
    }
    session->provider = provider;
    *session_out = session;
    return status;
}


void openvino_device(GLiClassDevice device, char** device_type) {
    switch (device) {
    case GC_NPU:
        *device_type = (char*)calloc(4, sizeof(char));
        snprintf(*device_type, 4, "NPU");
        return;
    case GC_CPU:
        *device_type = (char*)calloc(4, sizeof(char));
        snprintf(*device_type, 4, "CPU");
        return;
    case GC_GPU:
        *device_type = (char*)calloc(4, sizeof(char));
        snprintf(*device_type, 4, "GPU");
        return;
    default: // GPU with id
        *device_type = (char*)calloc(6, sizeof(char));
        snprintf(*device_type, 6, "GPU.%d", device);
    }
    return;
}


int get_cuda_device_id(GLiClassDevice device) {
    if (device == GC_GPU) return 0;
    return device;
}


GLiClassStatus* gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const bool use_mutex,
    const GLiClassProvider provider,
    const GLiClassDevice device,
    GLiClassSession** session_out
) {
    GLiClassProviderAPI* provider_api = NULL;
    GLiClassStatus* status;
    if (provider == GC_ONNX) {
        #ifdef GC_USE_ONNX
        gliclass_ort_initialize_api();
        if (!g_ort) {
            return set_error(GC_PROVIDER_ERROR, "Unable to init ORT API");
        }

        if (device >= GC_GPU && device <= GC_GPU_8) {
            #ifdef GC_USE_CUDA
            // TODO: select device with id
            status = gliclass_ort_cuda_init(
                model_path, num_threads, get_cuda_device_id(device), &provider_api
            );
            if (status != NULL) return status;
            #else
            return set_error(GC_PROVIDER_ERROR, "ONNX CUDA provider is not supported for current build");
            #endif
        } else if (device == GC_CPU) {
            status = gliclass_ort_cpu_init(model_path, num_threads, &provider_api);
            if (status != NULL) return status;
        } else {
            return set_error(GC_LOGICAL_ERROR, "Device type is not supported for current build");
        }
        #else
        return set_error(GC_PROVIDER_ERROR, "ONNX provider is not supported for current build");
        #endif
    } else if (provider == GC_ONNX_OPENVINO) {
        #ifdef GC_USE_ONNX
        gliclass_ort_initialize_api();
        if (!g_ort) {
            return set_error(GC_PROVIDER_ERROR, "Unable to init ORT API");
        }

        char* device_type = NULL;
        openvino_device(device, &device_type);
        status = gliclass_ort_openvino_init(model_path, num_threads, device_type, &provider_api);
        free(device_type);
        if (status != NULL) return status;
        #else
        return set_error(GC_PROVIDER_ERROR, "ONNX OpenVino provider is not supported for current build");
        #endif
    } else if (provider == GC_OPENVINO) {
        #ifdef GC_USE_OPENVINO
        char* device_type = NULL;
        openvino_device(device, &device_type);
        status = gliclass_openvino_init(model_path, num_threads, device_type, &provider_api);
        free(device_type);
        if (status != NULL) return status;
        #else
        return set_error(GC_PROVIDER_ERROR, "OpenVino provider is not supported for current build");
        #endif
    } else {
        return set_error(GC_PROVIDER_ERROR, "Provider is not supported for current build");
    }
    return gliclass_init_custom_provider(
        model_config_path, tokenizer_path, use_mutex, provider_api, session_out
    );
}


GLiClassStatus* gliclass_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results,
    GLiClassTokensInfo* info
) {
    if (!session || !input_text || !labels || num_labels == 0) {
        return set_error(GC_LOGICAL_ERROR, "Inputs have invalid value!");
    }

    // Allocate output array for results
    *out_num_results = 0;
    *out_results = (GLiClassResult*)calloc(num_labels, sizeof(GLiClassResult));
    if (!(*out_results)) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate results");
    }

    char* input = NULL;
    GLiClassStatus* status = prepare_input(
        input_text, 
        labels, 
        num_labels, 
        session->model_config->prompt_first, 
        config->add_prefix_space,
        &input
    );
    if (status != NULL) {
        return set_error(GC_INFERENCE_ERROR, "Error while preparing text");
    }

    TokenizedInput tokenized;
    status = tokenize_input(
        session->tokenizer, 
        (const char*)input,
        config->min_length,
        config->max_length,
        &tokenized,
        info
    );

    if (status != NULL) {
        free(input);
        return status;
    }

    if (tokenized.seq_length == 0) {
        free_tokenized_input(&tokenized);
        free(input);
        return NULL;
    }
    status = session->provider->run_inference(
        session, config, &tokenized, labels, num_labels, out_results, out_num_results
    );
    free_tokenized_input(&tokenized);
    free(input);
    return status;
}


size_t get_batch_size(
    const size_t batch_idx,
    const size_t num_batches, 
    const size_t num_texts, 
    const size_t batch_size
) {
    return (batch_idx == num_batches - 1) ? (num_texts - batch_idx * batch_size) : batch_size;
}


GLiClassStatus* gliclass_infer_batch( // TODO: add quick exit on empty batches
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* input_texts[],
    const size_t num_texts,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult** out_results[],
    size_t* out_num_results[],
    size_t* out_num_results_size,
    GLiClassTokensInfo** info
) {
    if (
        !session || !input_texts || !labels || !num_labels || num_labels_size == 0 
        || (num_labels_size != 1 && num_labels_size != num_texts)
    ) {
        return set_error(GC_LOGICAL_ERROR, "Inputs have invalid value!");
    }

    *out_num_results_size = num_texts;
    *out_results = (GLiClassResult**)calloc(*out_num_results_size, sizeof(GLiClassResult*));
    *out_num_results = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!out_num_results || !out_results) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate results");
    }

    // Allocate memory for tensors
    size_t num_batches = (num_texts + config->batch_size - 1) / config->batch_size;
    bool same_labels = num_labels_size == 1;
    GLiClassStatus* status = NULL;

    #pragma omp parallel for schedule(dynamic) shared(status)
    for (size_t i = 0; i < num_batches; i++) {
        if (status != NULL) continue;
        
        // Prepare input data
        GLiClassStatus* status_in = NULL;
        const size_t current_batch_size = get_batch_size(
            i, num_batches, num_texts, config->batch_size
        );
        const char** batch_texts = input_texts + i*config->batch_size;
        const char*** batch_labels = same_labels ? labels : labels + i*config->batch_size;
        const size_t* batch_num_labels = same_labels ? num_labels : num_labels + i*config->batch_size;
        const size_t batch_num_labels_size = same_labels ? num_labels_size: current_batch_size;

        // Prepare tokens
        char** prepared_inputs = NULL;
        status_in = prepare_inputs(
            session->model_config,
            config,
            batch_texts,
            current_batch_size,
            batch_labels, 
            batch_num_labels, 
            same_labels,
            &prepared_inputs
        );
        if (status_in  != NULL) {
            status = status_in;
            continue;
        };

        TokenizedInputs tokenized;
        status_in = tokenize_inputs(
            session->tokenizer, 
            (const char**)prepared_inputs, 
            current_batch_size,
            config->min_length,
            config->max_length,
            &tokenized,
            info
        );
        if (status_in  != NULL) {
            status = status_in;
            free_prepared_inputs(prepared_inputs, current_batch_size);
            continue;
        };

        status_in = session->provider->run_inference_batch(
            session, config, &tokenized, i, batch_labels, batch_num_labels, batch_num_labels_size, 
            out_results, out_num_results
        );

        // Clean up memory
        free_prepared_inputs(prepared_inputs, current_batch_size);
        free_tokenized_inputs(&tokenized);
        if (status_in  != NULL) {
            status = status_in;
        }
    }

    if (status != NULL) {
        gliclass_free_results_batch(*out_results, *out_num_results, *out_num_results_size);
        *out_num_results_size = 0;
    }
    return status;
}


void gliclass_cleanup(GLiClassSession* session) {
    if (!session) return;
    if (session->use_mutex) {
        free_mutex();
    }
    if (session->model_config) free(session->model_config);
    if (session->tokenizer) tokenizers_free(session->tokenizer);
    gliclass_cleanup_provider(session->provider);
    free(session);
}


void gliclass_cleanup_provider(GLiClassProviderAPI* provider) {
    if (!provider) return;
    provider->cleanup(provider->session);
    free(provider);
}