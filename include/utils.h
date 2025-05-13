#ifndef CONFIGS_H
#define CONFIGS_H

#include <stdio.h>

// Define the messages
#define DONE_MSG  "\033[1;32m[== DONE ==]\033[0m"
#define ERROR_MSG "\033[1;31m[== ERROR ==]\033[0m"

void print_done(const char *format, ...);
void print_error(const char *format, ...); 

#endif // CONFIGS_H