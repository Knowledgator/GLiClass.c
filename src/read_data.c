#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cJSON.h" 

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
void parse_json(const char* json_string, char*** texts, size_t* num_texts, char**** labels,
                size_t** num_labels, size_t* num_labels_size, bool* same_labels, char** classification_type) {
    // Parse json
    cJSON* json = cJSON_Parse(json_string);
    if (!json) {
        fprintf(stderr, "Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        return;
    }
    
    // Get array texts
    cJSON* texts_json = cJSON_GetObjectItemCaseSensitive(json, "texts");
    if (cJSON_IsArray(texts_json)) {
        *num_texts = cJSON_GetArraySize(texts_json);
        *texts = (char**)malloc(*num_texts * sizeof(char*));
        for (size_t i = 0; i < *num_texts; ++i) {
            cJSON* text = cJSON_GetArrayItem(texts_json, i);
            if (cJSON_IsString(text)) {
                (*texts)[i] = strdup(text->valuestring);
            }
        }
    }
    // get value classification_type
    cJSON* classification_type_json = cJSON_GetObjectItemCaseSensitive(json, "classification_type");
    if (cJSON_IsString(classification_type_json)) {
        *classification_type = strdup(classification_type_json->valuestring);
    }

    // get value same_labels
    cJSON* same_labels_json = cJSON_GetObjectItemCaseSensitive(json, "same_labels");
    if (cJSON_IsBool(same_labels_json)) {
        *same_labels = cJSON_IsTrue(same_labels_json);
    }
    
    if (*same_labels){
        // Get array labels
        cJSON* labels_json = cJSON_GetObjectItemCaseSensitive(json, "labels");
        if (cJSON_IsArray(labels_json)) {
            *num_labels_size = cJSON_GetArraySize(labels_json);
            *labels = (char**)malloc(*num_labels_size * sizeof(char*));
            for (size_t i = 0; i < *num_labels_size; ++i) {
                cJSON* label = cJSON_GetArrayItem(labels_json, i);
                if (cJSON_IsString(label)) {
                    (*labels)[i] = strdup(label->valuestring);
                }
            }
        }
        // Dynamically create an array num_labels corresponding to the number of texts
        *num_labels = (size_t*)malloc(*num_texts * sizeof(size_t));
        for (size_t i = 0; i < *num_texts; ++i) {
            (*num_labels)[i] = *num_labels_size;  // Number of labels for each text
        }
    } else {
        // We get an array of labels for each text (array of arrays)
        cJSON* labels_json = cJSON_GetObjectItemCaseSensitive(json, "labels");
        if (cJSON_IsArray(labels_json)) {
            // We check that the number of tags matches the number of texts
            if (cJSON_GetArraySize(labels_json) != *num_texts) {
                fprintf(stderr, "Error:the number of arrays of labels does not match the number of texts.\n");
                cJSON_Delete(json);
                return;
            }

            *num_labels = (size_t*)malloc(*num_texts * sizeof(size_t)); // dynamic array num_labels
            *labels = (char***)malloc(*num_texts * sizeof(char**));     // array of arrays for each group of labels
            
            // We iterate over each text
            for (size_t i = 0; i < *num_texts; ++i) {
                cJSON* text_labels_json = cJSON_GetArrayItem(labels_json, i);
                if (cJSON_IsArray(text_labels_json)) {
                    size_t num_labels_for_text = cJSON_GetArraySize(text_labels_json);
                    (*num_labels)[i] = num_labels_for_text;
                    (*labels)[i] = (char**)malloc(num_labels_for_text * sizeof(char*));
                    if (!(*labels)[i]) {
                        fprintf(stderr, "Error: failed to allocate memory for text labels %zu.\n", i);
                        cJSON_Delete(json);
                        return;
                    }
                    for (size_t j = 0; j < num_labels_for_text; ++j) {
                        cJSON* label_item  = cJSON_GetArrayItem(text_labels_json, j);
                        if (cJSON_IsString(label_item)) {
                            (*labels)[i][j] = strdup(label_item->valuestring);
                        }
                    }
                }else{
                    fprintf(stderr, "Error: labels forr text %zu are not array.\n", i);
                }
            }
        }
    }
    cJSON_Delete(json);  // free memory
}

/**
 * Converts a string to a boolean value.
 * 
 * Accepts the following string values:
 * - "true" or "1" will return true
 * - "false" or "0" will return false
 * 
 * If the string does not match these values, the function prints an error message
 * and exits the program with code 1.
 *
 * @param str The string to convert to a boolean. Expected values are "true", "false", "1", or "0".
 * @return true if the input is "true" or "1", false if the input is "false" or "0".
 */
bool string_to_bool(const char *str) {
    if (strcmp(str, "true") == 0 || strcmp(str, "1") == 0) {
        return true;
    } else if (strcmp(str, "false") == 0 || strcmp(str, "0") == 0) {
        return false;
    } else {
        printf("Invalid value for bool argument. Use 'true' or 'false'.\n");
        exit(1);
    }
}