#include "read_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cJSON.h"

#include "error.h"

GLiClassStatus* read_file(const char* filename, char** content) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return set_error(GC_LOGICAL_ERROR, "Error: Failed to open file %s\n", filename);
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    *content = (char*)calloc(length + 1, sizeof(char));
    if (!(*content)) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate model config");
    }
    fread(*content, sizeof(char), length, file);
    (*content)[length] = '\0';
    fclose(file);
    return NULL;
}


GLiClassStatus* parse_model_config_json(const char* json_string, GLiClassModelConfig** config_out) {
    GLiClassModelConfig* config = (GLiClassModelConfig*)calloc(1, sizeof(GLiClassModelConfig));
    if (!config) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate model config");
    }

    // Parse json
    cJSON* json = cJSON_Parse(json_string);
    if (!json) {
        free(config);
        return set_error(GC_FILE_ERROR, "Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
    }
    
    // Get array texts
    cJSON* prompt_first_field = cJSON_GetObjectItemCaseSensitive(json, "prompt_first");
    if (prompt_first_field && cJSON_IsBool(prompt_first_field)) {
        config->prompt_first = cJSON_IsTrue(prompt_first_field);
    } else {
        free(config);
        cJSON_Delete(json);
        return set_error(GC_FILE_ERROR, "Unexpected config format, expected 'prompt_first' field of bool type");
    }
    
    cJSON_Delete(json);  // free memory
    *config_out = config;
    return NULL;
}