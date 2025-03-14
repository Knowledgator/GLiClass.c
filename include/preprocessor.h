#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stddef.h>
#include <stdbool.h>
#include "gliclass_api.h"

const char** prepare_inputs(
    GLiClassSession* session,
    const char* texts[], 
    const size_t num_texts,
    const char** labels[], 
    const size_t num_labels[],
    bool same_labels
);
                    
char* prepare_input(
    const char* text, 
    const char** labels, 
    size_t num_labels, 
    bool prompt_first,
    bool append_prefix_space
);
void free_prepared_inputs(char** prepared_inputs, size_t num_texts);

#endif // PREPROCESSOR_H
