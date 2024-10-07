#ifndef READ_DATA_H
#define READ_DATA_H

#include <stddef.h>
#include <stdbool.h>

char* read_file(const char* filename);
void parse_json(const char* json_string, char*** texts, size_t* num_texts, char**** labels,
                size_t** num_labels, size_t* num_labels_size, bool* same_labels, char** classification_type); 
bool string_to_bool(const char *str);
#endif // READ_DATA_H