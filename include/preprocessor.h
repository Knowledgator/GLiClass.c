#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stddef.h>
#include <stdbool.h>

const char** prepare_inputs(const char* texts[], const char** const* labels, size_t num_texts,
                    size_t num_labels[], bool same_labels, bool prompt_first);
                    
char* prepare_input(const char* text, const char* labels[], size_t num_labels, bool prompt_first);
void free_prepared_inputs(char** prepared_inputs, size_t num_texts);

#endif // PREPROCESSOR_H
