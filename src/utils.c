#include "utils.h"

#include <stdio.h>
#include "read_data.h"

GLiClassModelConfig* initialize_model_config(const char* model_config_path) {
    const char* json_string = read_file(model_config_path);
    GLiClassModelConfig* c = parse_model_config_json(json_string);
    free((void*)json_string);
    return c;
}

int64_t* flatten_int_array(int64_t** data, size_t rows, size_t cols) {
    int64_t* flat_data = (int64_t*)calloc(rows * cols, sizeof(int64_t));
    if (!flat_data) {
        fprintf(stderr, "Error: Memory allocation for flat_data failed\n");
        return NULL;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_data[i * cols + j] = data[i][j];
        }
    }
    return flat_data;
}


float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


void process_multi_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    for (size_t j = 0; j < num_classes; j++) {
        float logit = output_data[j];
        float prob = sigmoid(logit);  // sigmoid function

        if (prob < threshold) continue;

        const char* label = NULL;
        if (j < num_labels) {
            label = labels[j];
        }
    
        if (!label) label = "[Unknown]";
        out_results[*out_num_results] = (GLiClassResult){(char*)label, prob};
        *out_num_results += 1;
    }
}

void process_multi_label_batch(
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
        out_results[text_id+i] = (GLiClassResult*)calloc(num_classes, sizeof(GLiClassResult));
        if (!out_results[text_id+i]) {
            fprintf(stderr, "Unable to allocate results");    
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
        process_multi_label(
            output_data+i*num_classes, // slide to current batch
            num_classes,
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
}

void process_single_label(
    const float* const output_data,
    const size_t num_classes,
    const char* labels[],
    const size_t num_labels,
    const float threshold,
    GLiClassResult out_results[],
    size_t* out_num_results
) {
    float max_prob = 0.0f; // TODO: skip first class
    size_t max_idx = 0;
    for (size_t j = 0; j < num_classes; j++) {
        float logit = output_data[j];
        float prob = sigmoid(logit);  // sigmoid function
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = j;
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

void process_single_label_batch(
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
            fprintf(stderr, "Unable to allocate results");    
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
        process_single_label(
            output_data+i*num_classes, // slide to current batch
            num_classes,
            current_labels,
            current_labels_size,
            threshold,
            out_results[text_id+i], // slide to current batch
            out_num_results + (text_id+i) // slide to current batch
        );
    }
}