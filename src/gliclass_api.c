#include "gliclass_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "model.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "parallel_processor.h"

#ifndef _WIN32
    #include <mqueue.h>
    #include <pthread.h>
#else
    #include <windows.h>
#endif

// Mutex declarations
#ifndef _WIN32
pthread_mutex_t queue_mutex;
#else
HANDLE queue_mutex;
#endif

const OrtApi* g_ort = NULL;

bool validate_inference_config(const InferenceConfig* inference_config) {
    char* template = "Inference paramete is invalid: %s";
    if (inference_config->batch_size == 0) {
        fprintf(stderr, template, "batch_size shouldn't equal zero");
        return false;
    } else if (inference_config->max_length == 0) {
        fprintf(stderr, template, "max_length shouldn't equal zero");
        return false;
    } else if (inference_config->cpu_threads == 0) {
        fprintf(stderr, template, "cpu_threads shouldn't equal zero");
        return false;
    } else if (inference_config->threshold < 0 && inference_config->threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (inference_config->threshold < 0 && inference_config->threshold > 1) {
        fprintf(stderr, template, "threshold should be in range 0 ... 1");
        return false;
    } else if (!(
        strcmp(inference_config->classification_type, "multi-label") 
        || strcmp(inference_config->classification_type, "single-label")
    )) {
        fprintf(stderr, template, "classification_type should be equal to 'multi-label' or 'single-label'");
        return false;
    }
    return true;
}

// TODO: add overload with inferance config + validate it + parse model config 
GLiClassSession* gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path, 
    const InferenceConfig* inference_config
) {
    // Initializes the ONNX Runtime API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) return NULL;

    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) return NULL;

    if (!validate_inference_config(inference_config)) {
        return NULL;
    }
    session->inference_config = inference_config;

    // Initialize the model config
    session->model_config = initialize_model_config(model_config_path);
    if (!session->model_config) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize ONNX environment
    session->env = initialize_ort_environment();
    if (!session->env) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize tokenizer
    session->tokenizer = create_tokenizer(tokenizer_path);
    if (!session->tokenizer) {
        gliclass_cleanup(session);
        return NULL;
    }

    // Initialize ONNX session (model loading)
    session->session = create_ort_session(
        session->env, model_path, session->inference_config->cpu_threads
    );
    if (!session->session) {
        gliclass_cleanup(session);
        return NULL;
    }

    return session;
}

// // single process
// bool gliclass_infer(
//     GLiClassSession* session,
//     const char* input_text,
//     const char* labels[],
//     const size_t num_labels,
//     GLiClassResult* out_results[],
//     size_t* out_num_results
// ) {
//     if (!session || !input_text || !labels || num_labels == 0) return false;

//     // Allocate output array for results
//     *out_num_results = num_labels;
//     *out_results = (GLiClassResult*)calloc(*out_num_results, sizeof(GLiClassResult));
//     if (!*out_results) {
//         fprintf(stderr, "Unable to allocate results");
//         return false;
//     }

//     char* input = prepare_input(input_text, labels, num_labels, session->model_config->prompt_first);
//     if (!input) {
//         fprintf(stderr, "Error while preparing text");
//         free(input);
//         return false;
//     }

//     TokenizedInput tokenized = tokenize_input(
//         session->tokenizer, 
//         (const char**)input, 
//         session->inference_config->max_length
//     );

//     OrtValue* input_ids_tensor = create_tensor(tokenized.input_ids, 1, tokenized.seq_length);
//     if (!input_ids_tensor) {
//         return false;
//     }
//     OrtValue* attention_mask_tensor = create_tensor(tokenized.attention_mask, 1, tokenized.seq_length);
//     if (!attention_mask_tensor) {
//         g_ort->ReleaseValue(input_ids_tensor);
//         return -1;
//     }
//     OrtValue* output_tensor = run_inference(session->session, input_ids_tensor, attention_mask_tensor);

//     process_output_tensor(
//         session,
//         output_tensor, 
//         labels, 
//         &num_labels, 
//         1, 
//         0, 
//         &out_results,
//         &out_num_results
//     );

//     return true;
// }


// batch process
bool gliclass_infer_batch(
    GLiClassSession* session,
    const char* input_texts[],
    const size_t num_texts,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult** out_results[],
    size_t* out_num_results[],
    size_t* out_num_results_size
) {
    if (!session || !input_texts || !labels || !num_labels || num_labels_size == 0) return false;

    // Initialize queue mutex
    #ifndef _WIN32
    pthread_mutex_init(&queue_mutex, NULL);
    #else
    queue_mutex = CreateMutex(NULL, FALSE, NULL);
    #endif

    *out_num_results_size = num_texts;
    *out_results = (GLiClassResult**)calloc(*out_num_results_size, sizeof(GLiClassResult*));
    *out_num_results = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!out_num_results || !out_results) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }

    // Allocate memory for tensors
    size_t num_batches = (
        num_texts + session->inference_config->batch_size - 1
    ) / session->inference_config->batch_size;
    OrtValue** input_ids_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** attention_mask_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));
    OrtValue** output_tensors = (OrtValue**)calloc(num_batches, sizeof(OrtValue*));

    // Preprocessing stage
    parallel_preprocess(
        session,
        num_batches,
        input_texts, 
        num_texts,
        labels, 
        num_labels, 
        num_labels_size, 
        input_ids_tensors,
        attention_mask_tensors
    );

    // Inference stage
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        #ifdef USE_CUDA // GPU
        pthread_mutex_lock(&queue_mutex);
        output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
        pthread_mutex_unlock(&queue_mutex);
        #else
        output_tensors[i] = run_inference(session->session, input_ids_tensors[i], attention_mask_tensors[i]);
        #endif
    }

    // Postprocess stage - processing batches
    parallel_postprocess(
        session,
        output_tensors, 
        num_batches,
        num_texts,
        labels, 
        num_labels,
        num_labels_size, 
        session->inference_config->classification_type,
        *out_results,
        *out_num_results
    );

    #ifndef _WIN32
	pthread_mutex_destroy(&queue_mutex);
    #else
	CloseHandle(queue_mutex);
    #endif
    return true;
}

bool gliclass_infer(
    GLiClassSession* session,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
) {
    if (!session || !input_text || !labels || num_labels == 0) return false;

    size_t out_num_results_size_tmp = 0;
    GLiClassResult** out_results_tmp = (GLiClassResult**)calloc(1, sizeof(GLiClassResult*));
    size_t* out_num_results_tmp = (size_t*)calloc(1, sizeof(size_t));
    if (!out_num_results_tmp || !out_results_tmp) {
        fprintf(stderr, "Unable to allocate results");
        return false;
    }
    bool ok = gliclass_infer_batch(session, &input_text, 1, &labels, &num_labels, 1, &out_results_tmp, &out_num_results_tmp, &out_num_results_size_tmp);

    if (ok) {
        *out_results = out_results_tmp[0];
        *out_num_results = out_num_results_tmp[0];
    }
    free(out_results_tmp);
    free(out_num_results_tmp);
    return ok;
}

void gliclass_free_results(GLiClassResult* results, size_t num_results) {
    if (!results) return;
    // for (size_t i = 0; i < num_results; i++) {
    //     free(results[i].label);
    // }
    free(results);
}

void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size) {
    if (!results) return;
    for (size_t i = 0; i < num_results_size; i++) {
        // for (size_t j = 0; j < num_results[i]; j++) {
        //     free(results[i][j].label);
        // }
        free(results[i]);
    }
    free(results);
}

void gliclass_cleanup(GLiClassSession* session) {
    if (!session) return;
    if (session->tokenizer) tokenizers_free(session->tokenizer);
    if (session->env) g_ort->ReleaseEnv(session->env);
    if (session->session) g_ort->ReleaseSession(session->session);
    free(session);
}