#include <stdio.h>
#include <stdlib.h>

#include "attendance/config.h"
#include "attendance/matcher.h"

int main(int argc, char **argv) {
    const char *db_path = (argc > 1) ? argv[1] : "data/embeddings.bin";

    AppConfig config;
    config_load("config.yaml", &config);

    printf("=====================================================\n");
    printf("   EMBEDDINGS DATABASE INSPECTOR (C EDITION)         \n");
    printf("=====================================================\n");
    printf("Inspecting database: %s\n\n", db_path);

    FaceMatcher *matcher = face_matcher_create(db_path, &config.face);
    if (!matcher) {
        fprintf(stderr, "Failed to load matcher from %s\n", db_path);
        return 1;
    }

    int count = face_matcher_get_count(matcher);
    printf("Total Enrolled Templates: %d\n", count);

    face_matcher_destroy(matcher);
    return 0;
}
