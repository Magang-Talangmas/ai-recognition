#ifndef ATTENDANCE_MATCHER_H
#define ATTENDANCE_MATCHER_H

#include <stdbool.h>
#include "attendance/config.h"
#include "attendance/face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char employee_id[128];
    bool is_matched;
    float score;
    float margin;
} MatchResult;

typedef struct FaceMatcher FaceMatcher;

/* Create FaceMatcher and load template database from binary/npz/dat file */
FaceMatcher *face_matcher_create(const char *embedding_file, const FaceConfig *config);

/* Free FaceMatcher resources */
void face_matcher_destroy(FaceMatcher *matcher);

/* Match a query face embedding (512 float vector) against enrolled templates */
MatchResult face_matcher_match(const FaceMatcher *matcher, const float *embedding);

/* Get number of enrolled employee templates */
int face_matcher_get_count(const FaceMatcher *matcher);

/* Save enrolled templates into a binary embedding database file */
int face_matcher_save_database(
    const char *filepath,
    const char (*employee_ids)[128],
    const float *templates,
    int count
);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_MATCHER_H */
