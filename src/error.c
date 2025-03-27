#include "error.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

GLiClassError* last_error = NULL;

void set_error(const char* format, ...) {
    va_list args;
    va_start(args, format);

    if (last_error) {
        free(last_error);
    }
    last_error = (GLiClassError*)calloc(1, sizeof(GLiClassError));
    if (!last_error) return;
    snprintf(last_error->message, GLICLASS_STATUS_MESSAGE_SIZE, args);
}