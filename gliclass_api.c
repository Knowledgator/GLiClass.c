#include "gliclass_api.h"

#include "tokenizer.h"
#include "model.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "paths.h"
#include "configs.h"
#include <stdlib.h>
#include <string.h>

struct GLiClassSession {
    TokenizerHandle tokenizer;
    OrtSession* session;
    OrtEnv* env;
};

const OrtApi* g_ort = NULL; // Global pointer to ONNX Runtime API for performing model inference

GLiClassSession* gliclass_init(const char* model_path, const char* tokenizer_path, int num_threads) {
    GLiClassSession* session = calloc(1, sizeof(GLiClassSession));
    if (!session) return NULL;

    initialize_ort_api();
    if (!g_ort) {
        free(session);
        return NULL;
    }

    // Initialize ONNX environment
    session->env = initialize_ort_environment();
    if (!session->env) {
        free(session);
        return NULL;
    }

    // Initialize tokenizer
    session->tokenizer = create_tokenizer(tokenizer_path);
    if (!session->tokenizer) {
        g_ort->ReleaseEnv(session->env); // Clean env
        free(session);
        return NULL;
    }

    // Initialize ONNX session (model loading)
    session->session = create_ort_session(session->env, model_path, num_threads);
    if (!session->session) {
        tokenizers_free(session->tokenizer); // Free tokenizer
        g_ort->ReleaseEnv(session->env);     // Free env
        free(session);
        return NULL;
    }

    // Initialize random seed 
    srand((unsigned int)time(NULL));

    // Success
    return session;
}

bool gliclass_infer(
    GLiClassSession* session,
    const char* input_text,
    const char** labels,
    size_t num_labels,
    GLiClassResult** out_results,
    size_t* out_num_results
) {
    if (!session || !input_text || !labels || num_labels == 0) return false;

    // Allocate output array for results
    *out_num_results = num_labels;
    *out_results = (GLiClassResult*)calloc(*out_num_results, sizeof(GLiClassResult));
    if (!*out_results) return false;

    // Fill each label with a random score
    for (size_t i = 0; i < num_labels; ++i) {
        (*out_results)[i].label = strdup(labels[i]); // Duplicate label text
        (*out_results)[i].score = (float)rand() / (float)RAND_MAX; // Random score between 0.0 and 1.0
    }

    return true; // Mock success
}

void gliclass_free_results(GLiClassResult* results, size_t num_results) {
    if (!results) return;
    for (size_t i = 0; i < num_results; i++) {
        free(results[i].label);
    }
    free(results);
}

void gliclass_cleanup(GLiClassSession* session) {
    if (!session) return;

    if (session->tokenizer) tokenizers_free(session->tokenizer);
    if (session->session) g_ort->ReleaseSession(session->session);
    if (session->env) g_ort->ReleaseEnv(session->env);
    free(session);
}