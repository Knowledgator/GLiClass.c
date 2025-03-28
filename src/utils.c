#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "error.h"

// Mutex declarations
#ifndef _WIN32
static pthread_mutex_t queue_mutex;
#else
static HANDLE queue_mutex;
#endif

void init_mutex() {
    // Initialize queue mutex
    #ifndef _WIN32
    pthread_mutex_init(&queue_mutex, NULL);
    #else
    queue_mutex = CreateMutex(NULL, FALSE, NULL);
    #endif
}

void lock_mutex() {
    #ifndef _WIN32
    pthread_mutex_lock(&queue_mutex);
    #else
    WaitForSingleObject(queue_mutex, INFINITE); 
    #endif
}

void unlock_mutex() {
    #ifndef _WIN32
    pthread_mutex_unlock(&queue_mutex);
    #else
    ReleaseMutex(queue_mutex);
    #endif
}

void free_mutex() {
    #ifndef _WIN32
    pthread_mutex_destroy(&queue_mutex);
    #else
    CloseHandle(queue_mutex);
    #endif
}

GLiClassStatus flatten_int_array(
    int64_t** data, size_t rows, size_t cols, int64_t** flat_data
) {
    *flat_data = (int64_t*)calloc(rows * cols, sizeof(int64_t));
    if (!(*flat_data)) {
        set_error("Error: Memory allocation for flat_data failed");
        return GC_MEMORY_ERROR;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            (*flat_data)[i * cols + j] = data[i][j];
        }
    }
    return GC_OK;
}


float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


void process_multi_label(
    const float* const output_data,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    for (size_t i = 0; i < num_labels; i++) {
        float logit = output_data[i];
        float prob = sigmoid(logit);  // sigmoid function

        if (prob < threshold) continue;

        out_results[*out_num_results] = (GLiClassResult){(char*)labels[i], prob};
        *out_num_results += 1;
    }
}

GLiClassStatus process_multi_label_batch(
    const float* const output_data,
    const size_t batch_size,
    const size_t num_classes,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size,
    const float threshold,
    const size_t text_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
) {
    bool same_labels = num_labels_size == 1;
    for (size_t i = 0; i < batch_size; i++) {
        
        const char** current_labels = NULL;
        size_t current_labels_size;
        if (same_labels) {
            current_labels = labels[0];
            current_labels_size = num_labels[0];
        } else {
            current_labels = labels[i];
            current_labels_size = num_labels[i];
        }

        current_labels_size = current_labels_size > num_classes ? num_classes : current_labels_size;
        out_results[text_id+i] = (GLiClassResult*)calloc(current_labels_size, sizeof(GLiClassResult));
        if (!out_results[text_id+i]) {
            set_error("Unable to allocate results");    
            return GC_MEMORY_ERROR;
        }
        process_multi_label(
            output_data+i*num_classes, // slide to current batch
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
    return GC_OK;
}

void process_single_label(
    const float* const output_data,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    float max_prob = 0.0f; // TODO: skip first class
    size_t max_idx = 0;
    for (size_t i = 0; i < num_labels; i++) {
        float logit = output_data[i];
        float prob = sigmoid(logit);  // sigmoid function
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = i;
        }
    }
    if (max_prob < threshold) return;

    const char* label = NULL;
    if (max_idx < num_labels) {
        label = labels[max_idx];
    }

    if (!label) label = "[Unknown]";
    out_results[0] = (GLiClassResult){(char*)label, max_prob};
    *out_num_results += 1;
}

GLiClassStatus process_single_label_batch(
    const float* const output_data,
    const size_t batch_size,
    const size_t num_classes,
    const char*** labels,
    const size_t* num_labels,
    const size_t num_labels_size,
    const float threshold,
    const size_t text_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
) {
    bool same_labels = num_labels_size == 1;
    for (size_t i = 0; i < batch_size; i++) {
        out_results[text_id+i] = (GLiClassResult*)calloc(1, sizeof(GLiClassResult));
        if (!out_results[text_id+i]) {
            set_error("Unable to allocate results");    
            return GC_MEMORY_ERROR;
        }

        const char** current_labels = NULL;
        size_t current_labels_size;
        if (same_labels) {
            current_labels = labels[0];
            current_labels_size = num_labels[0];
        } else {
            current_labels = labels[i];
            current_labels_size = num_labels[i];
        }

        current_labels_size = current_labels_size > num_classes ? num_classes : current_labels_size;

        process_single_label(
            output_data+i*num_classes, // slide to current batch
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
    return GC_OK;
}