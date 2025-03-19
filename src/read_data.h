#ifndef READ_DATA_H
#define READ_DATA_H

#include <stddef.h>
#include <stdbool.h>
#include "GLiClass/gliclass_api.h"

char* read_file(const char* filename);
GLiClassModelConfig* parse_model_config_json(const char* json_string); 
bool string_to_bool(const char *str);
#endif // READ_DATA_H