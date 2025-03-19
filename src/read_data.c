#include "read_data.h"

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


GLiClassModelConfig* parse_model_config_json(const char* json_string) {
    GLiClassModelConfig* config = (GLiClassModelConfig*)calloc(1, sizeof(GLiClassModelConfig));
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