#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cJSON.h" 

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