#include "preprocessor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

const char** prepare_inputs(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* texts[], 
    const size_t num_texts,
    const char** labels[], 
    const size_t num_labels[],
    const bool same_labels
){
    // Array to store prepared data
    char** inputs = (char**)calloc(num_texts, sizeof(char*));
    if (!inputs) {
        fprintf(stderr, "Error: cant allocate memory for array inputs\n");
        return NULL;
    }

    for (size_t i = 0; i < num_texts; ++i) {
        if (same_labels){
            inputs[i] = prepare_input(texts[i], labels[0], num_labels[0], session->model_config->prompt_first, config->add_prefix_space);
        } else {
            inputs[i] = prepare_input(texts[i], labels[i], num_labels[i], session->model_config->prompt_first, config->add_prefix_space);
        }

        if (!inputs[i]) {
            fprintf(stderr, "Error while preparing text for text: %zu\n", i);
            for (size_t j = 0; j < i; ++j) {
                free(inputs[j]);
            }
            free(inputs);
            return NULL;
        }
    }
    return (const char**)inputs;
}


void append_label(const char* label, char* result) {
    // add label in lower case
    for (const char* p = label; *p; ++p) {
        char lower_char = tolower((unsigned char)*p);
        strncat(result, &lower_char, 1);
    }
}


void append_labels(
    const char* label_prefix, 
    const char** labels, 
    const size_t num_labels, 
    char* result
) {
    for (size_t i = 0; i < num_labels; ++i) {
        strcat(result, label_prefix);
        append_label(labels[i], result);
    }
}


char* prepare_input(
    const char* text, 
    const char** labels,
    size_t num_labels,
    bool prompt_first,
    bool add_prefix_space
){
    const char* label_prefix = add_prefix_space ? "<<LABEL>> " : "<<LABEL>>";
    const char* sep_tag = "<<SEP>>";
    size_t total_len = (
        strlen(text) + strlen(sep_tag) + (add_prefix_space ? 2 : 1) + num_labels*strlen(label_prefix)
    ); // +1 for null terminator and +1 for space

    // size of result str
    for (size_t i = 0; i < num_labels; ++i) {
        total_len += strlen(labels[i]);
    }    

    char* result = (char*)calloc(total_len, sizeof(char));
    if (!result) {
        fprintf(stderr, "Cant allocate memmory for result prepared string\n");
        return NULL;
    }

    if (prompt_first) {
        append_labels(label_prefix, labels, num_labels, result);
        strcat(result, sep_tag);
        if (add_prefix_space) strcat(result, " ");
        strcat(result, text);
    } else {
        if (add_prefix_space) strcat(result, " ");
        strcat(result, text);
        append_labels(label_prefix, labels, num_labels, result);
        strcat(result, sep_tag);
    }

    return result;
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