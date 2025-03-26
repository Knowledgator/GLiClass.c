#include "error.h"

#include <stdarg.h>

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