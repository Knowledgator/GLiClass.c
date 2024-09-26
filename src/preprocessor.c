#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#include "preprocessor.h"

/**
 * Prepares inputs for further processing by combining texts with their corresponding labels.
 * Each input will be formatted according to whether the labels should be included first (prompt_first)
 * and whether the same labels apply to all texts.
 *
 * @param texts An array of input texts.
 * @param labels An array of labels (or array of arrays if same_labels is false).
 * @param num_texts The number of texts to process.
 * @param num_labels An array of size_t, indicating the number of labels for each text.
 * @param same_labels If true, all texts share the same labels; otherwise, each text has its own labels.
 * @param prompt_first If true, labels are added before the text; otherwise, they are appended after the text.
 * @return A dynamically allocated array of strings, where each string contains the prepared input.
 *         The caller is responsible for freeing the memory.
 */
char** prepare_inputs(const char* texts[], const char* labels[], size_t num_texts,
                    size_t num_labels[], bool same_labels, bool prompt_first){

    // Array to store prepared data
    char** inputs = (char**)malloc(num_texts * sizeof(char*));
    
    if (!inputs) {
        fprintf(stderr, "Error: cant allocate memory for array inputs\n");
        return NULL;
    }    
    for (size_t i = 0; i < num_texts; ++i) {
        if (same_labels){
            inputs[i] = prepare_input(texts[i], labels, num_labels[0], prompt_first);
        } else {
            inputs[i] = prepare_input(texts[i], labels[i], num_labels[i], prompt_first);
        }

        if (!inputs[i]) {
            fprintf(stderr, "Error while preparing text for text: %zu\n", i);
            // Освобождение уже выделенной памяти
            for (size_t j = 0; j < i; ++j) {
                free(inputs[j]);
            }
            free(inputs);
            return NULL;
        }
        
    }

    return inputs;
}


/**
 * Prepares a single input by combining a text with its labels, following a specific format.
 * If prompt_first is true, the labels are prepended to the text, otherwise they are appended.
 * Labels are formatted in lower case and are prefixed by "<<LABEL>>".
 *
 * @param text The input text to be prepared.
 * @param labels An array of labels corresponding to the text.
 * @param num_labels The number of labels to process for the text.
 * @param prompt_first If true, labels are added before the text; otherwise, they are appended after the text.
 * @return A dynamically allocated string containing the prepared input. The caller is responsible for freeing the memory.
 */
char* prepare_input(const char* text, const char* labels[], size_t num_labels, bool prompt_first){
    const char* label_prefix = "<<LABEL>>";
    const char* sep_tag = "<<SEP>>";
    size_t total_len = strlen(text) + strlen(sep_tag) + 1; // +1 for null terminator

    // size of result str
    for (size_t i = 0; i < num_labels; ++i) {
        total_len += strlen(label_prefix) + strlen(labels[i]);
    }    

    char* result = (char*)malloc(total_len * sizeof(char));
    if (!result) {
        fprintf(stderr, "Cant allocate memmory for result prepared string\n");
        return NULL;
    }

    result[0] = '\0'; // clear memory before use
    if (prompt_first) {
        for (size_t i = 0; i < num_labels; ++i) {
            strcat(result, label_prefix);

            // add label in lower case
            for (const char* p = labels[i]; *p; ++p) {
                char lower_char = tolower((unsigned char)*p);
                strncat(result, &lower_char, 1);
            }
        }
        strcat(result, sep_tag);
        strcat(result, text);
    } else {
        strcat(result, text);
        for (size_t i = 0; i < num_labels; ++i) {
            strcat(result, label_prefix);

            // Добавление label в нижнем регистре
            for (const char* p = labels[i]; *p; ++p) {
                char lower_char = tolower((unsigned char)*p);
                strncat(result, &lower_char, 1);
            }
        }
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