#include "preprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include "error.h"

GLiClassStatus* prepare_inputs(
    GLiClassModelConfig* model_config,
    const GLiClassInferenceConfig* config,
    const char* texts[], 
    const size_t num_texts,
    const char** labels[], 
    const size_t num_labels[],
    const bool same_labels,
    char*** inputs
) {
    // Array to store prepared data
    *inputs = (char**)calloc(num_texts, sizeof(char*));
    if (!(*inputs)) {
        return set_error(GC_MEMORY_ERROR, "Cant allocate memory for array inputs");
    }

    GLiClassStatus* status;
    for (size_t i = 0; i < num_texts; ++i) {
        if (same_labels){
            status = prepare_input(
                texts[i], labels[0], num_labels[0], model_config->prompt_first, config->add_prefix_space, &(*inputs)[i]
            );
        } else {
            status = prepare_input(texts[i], labels[i], num_labels[i], model_config->prompt_first, config->add_prefix_space, &(*inputs)[i]);
        }

        if (status != NULL) {
            for (size_t j = 0; j < i; ++j) {
                free(inputs[j]);
            }
            free(inputs);
            return status;
        }
    }
    return NULL;
}


void append_label(
    const char* label, const size_t result_size, char* result
) {
    // add label in lower case
    for (const char* p = label; *p; ++p) {
        char lower_char = (char)tolower((unsigned char)*p);
        strncat_s(result, result_size, &lower_char, 1);
    }
}


void append_labels(
    const char* label_prefix, 
    const char** labels, 
    const size_t num_labels, 
    const size_t result_size,
    char* result
) {
    for (size_t i = 0; i < num_labels; ++i) {
        strcat_s(result, result_size, label_prefix);
        append_label(labels[i], result_size, result);
    }
}


GLiClassStatus* prepare_input(
    const char* text, 
    const char** labels,
    size_t num_labels,
    bool prompt_first,
    bool add_prefix_space,
    char** input
) {
    const char* label_prefix = add_prefix_space ? "<<LABEL>> " : "<<LABEL>>";
    const char* sep_tag = "<<SEP>>";
    size_t total_len = (
        strlen(text) + strlen(sep_tag) + (add_prefix_space ? 2 : 1) + num_labels*strlen(label_prefix)
    ); // +1 for null terminator and +1 for space

    // size of result str
    for (size_t i = 0; i < num_labels; ++i) {
        total_len += strlen(labels[i]);
    }    

    *input = (char*)calloc(total_len, sizeof(char));
    if (!(*input)) {
        return set_error(GC_MEMORY_ERROR, "Unable allocate memmory for result prepared string");
    }

    if (prompt_first) {
        append_labels(
            label_prefix, labels, num_labels, total_len, *input
        );
        strcat_s(*input, total_len, sep_tag);
        if (add_prefix_space) strcat_s(*input, total_len, " ");
        strcat_s(*input, total_len, text);
    } else {
        if (add_prefix_space) strcat_s(*input, total_len, " ");
        strcat_s(*input, total_len, text);
        append_labels(
            label_prefix, labels, num_labels, total_len, *input
        );
        strcat_s(*input, total_len, sep_tag);
    }

    return NULL;
}

/**
 * Frees the memory allocated for the prepared inputs.
 *
 * @param prepared_inputs A dynamically allocated array of prepared input strings.
 * @param num_texts The number of input texts (size of the prepared_inputs array).
 */
void free_prepared_inputs(char** prepared_inputs, size_t num_texts){
    for (size_t i = 0; i < num_texts; i++){
        free(prepared_inputs[i]);
    }
    free(prepared_inputs);
}