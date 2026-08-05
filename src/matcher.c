#include "attendance/matcher.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MATCHER_MAGIC "FACES1\0\0"
#define MAX_ENROLLED_EMPLOYEES 1024

struct FaceMatcher {
    char employee_ids[MAX_ENROLLED_EMPLOYEES][128];
    float templates[MAX_ENROLLED_EMPLOYEES * FACE_EMBEDDING_DIM];
    int count;
    FaceConfig config;
};

FaceMatcher *face_matcher_create(const char *embedding_file, const FaceConfig *config) {
    FaceMatcher *matcher = (FaceMatcher *)calloc(1, sizeof(FaceMatcher));
    if (!matcher) return NULL;

    if (config) {
        matcher->config = *config;
    }

    const char *candidates[] = {
        embedding_file,
        "data/embeddings.bin",
        "data\\embeddings.bin",
        "../data/embeddings.bin",
        "camera-c/data/embeddings.bin",
        "D:/kamera/camera-c/data/embeddings.bin",
        "D:\\kamera\\camera-c\\data\\embeddings.bin",
        NULL
    };

    FILE *f = NULL;
    const char *found_path = NULL;

    for (int i = 0; candidates[i] != NULL; i++) {
        if (strlen(candidates[i]) == 0) continue;
        f = fopen(candidates[i], "rb");
        if (f) {
            found_path = candidates[i];
            break;
        }
    }

    if (!f) {
        printf("[Matcher] Info: Template file '%s' not found. Enroll faces to populate.\n",
               embedding_file ? embedding_file : "data/embeddings.bin");
        return matcher;
    }

    char magic[8] = {0};
    if (fread(magic, 1, 8, f) == 8 && memcmp(magic, "FACES1", 6) == 0) {
        int32_t count = 0;
        int32_t dim = 0;
        if (fread(&count, sizeof(int32_t), 1, f) == 1 &&
            fread(&dim, sizeof(int32_t), 1, f) == 1 && dim == FACE_EMBEDDING_DIM) {
            
            if (count > MAX_ENROLLED_EMPLOYEES) count = MAX_ENROLLED_EMPLOYEES;
            matcher->count = count;

            for (int i = 0; i < count; i++) {
                if (fread(matcher->employee_ids[i], 1, 128, f) != 128) break;
                if (fread(&matcher->templates[i * FACE_EMBEDDING_DIM], sizeof(float), FACE_EMBEDDING_DIM, f) != FACE_EMBEDDING_DIM) break;
                
                /* Ensure L2 normalized */
                face_vector_l2_normalize(&matcher->templates[i * FACE_EMBEDDING_DIM], FACE_EMBEDDING_DIM);
            }
        }
    } else {
        /* Fallback text loader for simple key-vector dumps */
        rewind(f);
        char line[4096];
        while (fgets(line, sizeof(line), f) && matcher->count < MAX_ENROLLED_EMPLOYEES) {
            char *comma = strchr(line, ',');
            if (!comma) continue;
            *comma = '\0';
            strncpy(matcher->employee_ids[matcher->count], line, 127);

            char *p = comma + 1;
            float *vec = &matcher->templates[matcher->count * FACE_EMBEDDING_DIM];
            for (int d = 0; d < FACE_EMBEDDING_DIM; d++) {
                vec[d] = (float)strtod(p, &p);
                if (*p == ',' || *p == ' ') p++;
            }
            face_vector_l2_normalize(vec, FACE_EMBEDDING_DIM);
            matcher->count++;
        }
    }

    fclose(f);
    printf("[Matcher] Loaded %d enrolled employee template(s) from '%s'\n", matcher->count, found_path);
    return matcher;
}

void face_matcher_destroy(FaceMatcher *matcher) {
    if (matcher) {
        free(matcher);
    }
}

int face_matcher_get_count(const FaceMatcher *matcher) {
    return matcher ? matcher->count : 0;
}

MatchResult face_matcher_match(const FaceMatcher *matcher, const float *embedding) {
    MatchResult result;
    memset(&result, 0, sizeof(MatchResult));
    result.score = 0.0f;
    result.margin = 0.0f;
    result.is_matched = false;

    if (!matcher || matcher->count == 0 || !embedding) {
        return result;
    }

    /* Normalize query vector copy */
    float query[FACE_EMBEDDING_DIM];
    memcpy(query, embedding, sizeof(query));
    face_vector_l2_normalize(query, FACE_EMBEDDING_DIM);

    int best_idx = -1;
    float best_score = -1.0f;
    float second_score = -1.0f;

    for (int k = 0; k < matcher->count; k++) {
        const float *tmpl = &matcher->templates[k * FACE_EMBEDDING_DIM];
        
        /* Fast Dot Product (Cosine Similarity on L2 normalized vectors) */
        float dot = 0.0f;
        for (int i = 0; i < FACE_EMBEDDING_DIM; i++) {
            dot += tmpl[i] * query[i];
        }

        if (dot > best_score) {
            second_score = best_score;
            best_score = dot;
            best_idx = k;
        } else if (dot > second_score) {
            second_score = dot;
        }
    }

    if (best_idx < 0) {
        return result;
    }

    result.score = best_score;
    result.margin = (second_score >= 0.0f) ? (best_score - second_score) : 1.0f;

    /* Verify thresholds */
    if (best_score >= matcher->config.match_threshold &&
        result.margin >= matcher->config.match_margin) {
        result.is_matched = true;
        strncpy(result.employee_id, matcher->employee_ids[best_idx], sizeof(result.employee_id) - 1);
    }

    return result;
}

int face_matcher_save_database(
    const char *filepath,
    const char (*employee_ids)[128],
    const float *templates,
    int count
) {
    if (!filepath || !employee_ids || !templates || count <= 0) return -1;

    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    fwrite("FACES1\0\0", 1, 8, f);
    int32_t c = (int32_t)count;
    int32_t d = (int32_t)FACE_EMBEDDING_DIM;
    fwrite(&c, sizeof(int32_t), 1, f);
    fwrite(&d, sizeof(int32_t), 1, f);

    for (int i = 0; i < count; i++) {
        fwrite(employee_ids[i], 1, 128, f);
        fwrite(&templates[i * FACE_EMBEDDING_DIM], sizeof(float), FACE_EMBEDDING_DIM, f);
    }

    fclose(f);
    return 0;
}
