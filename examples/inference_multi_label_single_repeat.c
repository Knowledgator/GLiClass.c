#include <stdio.h>
#include "GLiClass/gliclass_api.h"

int main() {
    const char* model_path = "./onnx/model.onnx";
    const char* model_config_path = "./onnx/config.json";
    const char* tokenizer_path = "./tokenizer/tokenizer.json";

    GLiClassInferenceConfig config;
    gliclass_create_inference_config(
        8, 2048, 0.5, "multi-label", true, &config
    );

    // Initialize session (model setup)
    GLiClassSession* session = gliclass_init(
        model_path,
        model_config_path,
        tokenizer_path,
        8
    );

    if (!session) {
        fprintf(stderr, "Failed to initialize GLiClass session!\n");
        return 1;
    }

    const char* texts[] = {
        "I'm in a reading rut and need recommendations about books! I typically enjoy fantasy sagas like Lord of the Rings or Game of Thrones, but I want something fresh. Any new releases worth checking out? I'll take anything with a well-built world and gripping plot. Help a fellow book lover out!",
        "I'm concerned about excessive automotive emissions. While carmakers like Toyota and Honda are spearheading the hybrid movement, the global shift to cleaner fuels isn't quick enough. Governments need to impose stricter regulations to curb pollution. Encouraging EV adoption can foster significant environmental benefits. Folks, what are your thoughts on this?"
    };
    const size_t num_texts = 2;

    const char* labels[] = {
        "Automobile",
        "Beauty",
        "Books",
        "Business",
        "Computers",
        "Education",
        "Electronics",
        "Entertainment",
        "Finance",
        "Fitness"
        /*
        "Food",
        "Games",
        "Government",
        "Health",
        "Hobbies",
        "Internet",
        "Jobs",
        "Law",
        "Leisure",
        "News",
        "Online Communities",
        "Pornography",
        "Real Estate",
        "Science",
        "Shopping",
        "Sports",
        "Telecom",
        "Travel"
        */
    };
    const size_t num_labels = 10;


    for (size_t t = 0; t < num_texts; ++t) {
        GLiClassResult* results = NULL;
        size_t num_results = 0;

        bool ok = gliclass_infer(
            session,
            &config,
            texts[t],
            labels,
            num_labels,
            &results,
            &num_results
        );

        if (!ok) {
            fprintf(stderr, "Errors occurred during inference on text %ld!\n", t + 1);
            gliclass_cleanup(session);
            return 1;
        }

        // Print results
        printf("\nText %ld: %s\n", t + 1, texts[t]);
        printf("num_results: %ld\n", num_results);
        for (size_t i = 0; i < num_results; ++i) {
            printf("Label_%ld: %s, Score: %f\n", i, results[i].label, results[i].score);
        }

        // Free results for this text
        gliclass_free_results(results, num_results);
    }

    gliclass_cleanup(session);
    return 0;
}