#include "read_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cJSON.h"
#include "gliclass_api.h"

/**
 * Reads the entire content of a file and returns it as a string.
 *
 * @param filename The name of the file to read.
 * @return A dynamically allocated string containing the file content, or NULL if the file could not be opened.
 *         The caller is responsible for freeing the allocated memory.
 */
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Faild to open file %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* content = (char*)malloc(length + 1);
    fread(content, 1, length, file);
    content[length] = '\0';
    fclose(file);
    return content;
}

/**
 * Parses a JSON string to extract information such as texts, labels, and classification type.
 *
 * @param json_string The JSON string to parse.
 * @param texts Pointer to an array of strings that will store the extracted texts.
 * @param num_texts Pointer to a size_t that will store the number of texts extracted.
 * @param labels Pointer to an array of arrays that will store the extracted labels for each text.
 * @param num_labels Pointer to a dynamic array of size_t representing the number of labels for each text.
 * @param num_labels_size Pointer to a size_t that will store the number of labels (if all texts have the same labels).
 * @param same_labels Pointer to a boolean that indicates whether all texts share the same labels.
 * @param classification_type Pointer to a string that will store the classification type.
 *
 * This function dynamically allocates memory for texts, labels, and related data. 
 * It is the caller's responsibility to free the allocated memory.
 */
ModelConfig* parse_model_config_json(const char* json_string) {
    ModelConfig* config = (ModelConfig*)calloc(1, sizeof(ModelConfig));
    if (!config) {
        return NULL;
    }

    // Parse json
    cJSON* json = cJSON_Parse(json_string);
    if (!json) {
        fprintf(stderr, "Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        return NULL;
    }
    
    // Get array texts
    cJSON* prompt_first_field = cJSON_GetObjectItemCaseSensitive(json, "prompt_first");
    if (prompt_first_field && cJSON_IsBool(prompt_first_field)) {
        config->prompt_first = cJSON_IsTrue(prompt_first_field);
    } else {
        fprintf(stderr, "Unexpected config format, expected 'prompt_first' field of bool type");
        return NULL;
    }
    
    cJSON_Delete(json);  // free memory
    return config;
}