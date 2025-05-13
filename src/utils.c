#include "utils.h"
#include <stdarg.h> 

void print_done(const char *format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stderr, "%s ", DONE_MSG);

    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");

    va_end(args);
}

void print_error(const char *format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stderr, "%s ", ERROR_MSG);

    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");

    va_end(args);
}