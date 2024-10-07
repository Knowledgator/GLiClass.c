#ifndef CONFIGS_H
#define CONFIGS_H

#define BATCH_SIZE 8    // Number of texts in one batch for processing by the model
#define MAX_LENGTH 2048 // Maximum length of tokenized text (number of tokens)
#define THRESHOLD 0.5f  // Threshold for making a classification decision 
#define NUM_THREADS 8   // Number of threads for CPU (does not affect GPU performance)

#endif // CONFIGS_H