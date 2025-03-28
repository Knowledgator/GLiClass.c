#include "GLiClass/gliclass_api.h"

#include "read_data.h"
#include "error.h"
#include "utils.h"
#include "preprocessor.h"
#include "tokenizer.h"

#include "GLiClass/gliclass.h"

GLiClassStatus initialize_model_config(const char* model_config_path, GLiClassModelConfig** model_config_out) {
    char* json_string = NULL;
    GLiClassStatus status = read_file(model_config_path, &json_string);
    if (status != GC_OK) return status;

    status = parse_model_config_json(json_string, model_config_out);
    free((void*)json_string);
    return status;
}

GLiClassStatus gliclass_init_custom_provider(
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const bool use_mutex,
    GLiClassProviderAPI* provider,
    GLiClassSession** session_out
) {
    if (num_threads == 0) {
        set_error("num_threads shouldn't equal zero");
        return GC_LOGICAL_ERROR;
    }

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) {
        set_error("Unable to allocate session");
        return GC_MEMORY_ERROR;
    };

    session->use_mutex = use_mutex;
    if (session->use_mutex) {
        init_mutex();
    }

    // Initialize the model config
    GLiClassStatus status = initialize_model_config(model_config_path, &(session->model_config));
    if (status != GC_OK) {
        gliclass_cleanup(session);
        return status;
    }

    // Initialize tokenizer
    status = create_tokenizer(tokenizer_path, &(session->tokenizer));
    if (status != GC_OK) {
        gliclass_cleanup(session);
        return status;
    }

    if (!provider) {
        set_error("No provider was provided");
        return GC_LOGICAL_ERROR;
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


GLiClassStatus gliclass_init(
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
    GLiClassStatus status;
    if (provider == GC_ONNX) {
        #ifdef USE_ONNX
        gliclass_ort_initialize_api();
        if (!g_ort) {
            set_error("Unable to init ORT API");
            return GC_PROVIDER_ERROR;
        }

        if (device >= GC_GPU && device <= GC_GPU_8) {
            #ifndef USE_CUDA
            // TODO: select device with id
            status = gliclass_ort_cuda_init(
                model_path, num_threads, get_cuda_device_id(device), &provider_api
            );
            if (status != GC_OK) return status;
            #else
            set_error("ONNX CUDA provider is not supported for current build");
            return GC_PROVIDER_ERROR;
            #endif
        } else if (device == GC_CPU) {
            status = gliclass_ort_cpu_init(model_path, num_threads, &provider_api);
            if (status != GC_OK) return status;
        } else {
            set_error("Device type is not supported for current build");
            return GC_LOGICAL_ERROR;
        }
        #else
        set_error("ONNX provider is not supported for current build");
        return GC_PROVIDER_ERROR;
        #endif
    } else if (provider == GC_ONNX_OPENVINO) {
        #ifdef USE_ONNX
        gliclass_ort_initialize_api();
        if (!g_ort) {
            set_error("Unable to init ORT API");
            return GC_PROVIDER_ERROR;
        }

        char* device_type = NULL;
        openvino_device(device, &device_type);
        status = gliclass_ort_openvino_init(model_path, num_threads, device_type, &provider_api);
        free(device_type);
        if (status != GC_OK) return status;
        #else
        set_error("ONNX OpenVino provider is not supported for current build");
        return GC_PROVIDER_ERROR;
        #endif
    } else if (provider == GC_OPENVINO) {
        #ifdef USE_OPENVINO
        char* device_type = NULL;
        openvino_device(device, &device_type);
        status = gliclass_openvino_init(model_path, num_threads, device_type, &provider_api);
        free(device_type);
        if (status != GC_OK) return status;
        #else
        set_error("OpenVino provider is not supported for current build");
        return GC_PROVIDER_ERROR;
        #endif
    } else {
        set_error("Provider is not supported for current build");
        return GC_PROVIDER_ERROR;
    }
    return gliclass_init_custom_provider(
        model_config_path, tokenizer_path, 
        num_threads, use_mutex, provider_api, session_out
    );
}


GLiClassStatus gliclass_infer(
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
        set_error("Inputs have invalid value!");
        return GC_LOGICAL_ERROR;
    }

    // Allocate output array for results
    *out_num_results = 0;
    *out_results = (GLiClassResult*)calloc(num_labels, sizeof(GLiClassResult));
    if (!(*out_results)) {
        set_error("Unable to allocate results");
        return GC_MEMORY_ERROR;
    }

    char* input = NULL;
    GLiClassStatus status = prepare_input(
        input_text, 
        labels, 
        num_labels, 
        session->model_config->prompt_first, 
        config->add_prefix_space,
        &input
    );
    if (status != GC_OK) {
        set_error("Error while preparing text");
        return GC_INFERENCE_ERROR;
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

    if (status != GC_OK) {
        free(input);
        return status;
    }

    if (tokenized.seq_length == 0) {
        free_tokenized_input(&tokenized);
        free(input);
        return GC_OK;
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


GLiClassStatus gliclass_infer_batch( // TODO: add quick exit on empty batches
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
        set_error("Inputs have invalid value!");
        return GC_LOGICAL_ERROR;
    }

    *out_num_results_size = num_texts;
    *out_results = (GLiClassResult**)calloc(*out_num_results_size, sizeof(GLiClassResult*));
    *out_num_results = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!out_num_results || !out_results) {
        set_error("Unable to allocate results");
        return GC_MEMORY_ERROR;
    }

    // Allocate memory for tensors
    size_t num_batches = (num_texts + config->batch_size - 1) / config->batch_size;
    bool same_labels = num_labels_size == 1;
    GLiClassStatus status = GC_OK;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        // Prepare input data
        const size_t current_batch_size = get_batch_size(
            i, num_batches, num_texts, config->batch_size
        );
        const char** batch_texts = input_texts + i*config->batch_size;
        const char*** batch_labels = same_labels ? labels : labels + i*config->batch_size;
        const size_t* batch_num_labels = same_labels ? num_labels : num_labels + i*config->batch_size;
        const size_t batch_num_labels_size = same_labels ? num_labels_size: current_batch_size;

        // Prepare tokens
        const char** prepared_inputs = NULL;
        status = prepare_inputs(
            session->model_config,
            config,
            batch_texts,
            current_batch_size,
            batch_labels, 
            batch_num_labels, 
            same_labels,
            &prepared_inputs
        );
        if (status != GC_OK) return status;

        TokenizedInputs tokenized;
        status = tokenize_inputs(
            session->tokenizer, 
            prepared_inputs, 
            current_batch_size,
            config->min_length,
            config->max_length,
            &tokenized,
            info
        );
        if (status != GC_OK) {
            free_prepared_inputs((char**)prepared_inputs, current_batch_size);
            return status;
        };

        status = session->provider->run_inference_batch(
            session, config, &tokenized, i, batch_labels, batch_num_labels, batch_num_labels_size, 
            out_results, out_num_results
        );

        // Clean up memory
        free_prepared_inputs((char**)prepared_inputs, current_batch_size);
        free_tokenized_inputs(&tokenized);
        if (status != GC_OK) break;
    }
    if (status != GC_OK) {
        gliclass_free_results_batch(*out_results, *out_num_results, *out_num_results_size);
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