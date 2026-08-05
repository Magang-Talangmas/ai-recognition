#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "attendance/config.h"
#include "attendance/database.h"
#include "attendance/face_engine.h"
#include "attendance/matcher.h"
#include "attendance/vision_utils.h"
#include "third_party/cJSON.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr, msg) \
    do { \
        if (expr) { \
            printf("  [PASS] %s\n", msg); \
            tests_passed++; \
        } else { \
            printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
            tests_failed++; \
        } \
    } while(0)

static void test_config(void) {
    printf("\n=== TEST 1: Config & Environment Expansion ===\n");
    AppConfig cfg;
    config_load(NULL, &cfg);

    ASSERT_TRUE(cfg.camera.process_every_n_frames == 10, "Default process_every_n_frames is 10");
    ASSERT_TRUE(cfg.face.vote_window == 5, "Default vote_window is 5");
    ASSERT_TRUE(cfg.face.votes_required == 3, "Default votes_required is 3");

    char expanded[256];
    config_expand_env("Hello ${PATH}", expanded, sizeof(expanded));
    ASSERT_TRUE(strlen(expanded) > 6, "Expanded ${PATH} environment variable");
}

static void test_matcher_and_normalization(void) {
    printf("\n=== TEST 2: Vector Normalization & Matcher Cosine Similarity ===\n");

    float vec[FACE_EMBEDDING_DIM];
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) vec[i] = (float)(i + 1);
    face_vector_l2_normalize(vec, FACE_EMBEDDING_DIM);

    double sum_sq = 0.0;
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) sum_sq += vec[i] * vec[i];
    ASSERT_TRUE(fabs(sum_sq - 1.0) < 1e-4, "L2 normalization yields unit length vector");

    /* Test template database save & load */
    char ids[2][128] = {"EMP-001", "EMP-002"};
    float tmpls[2 * FACE_EMBEDDING_DIM];
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) {
        tmpls[i] = (i == 0) ? 1.0f : 0.0f; /* 100% direction 0 */
        tmpls[FACE_EMBEDDING_DIM + i] = (i == 1) ? 1.0f : 0.0f; /* 100% direction 1 */
    }

    const char *test_db_file = "test_embeddings.bin";
    int save_rc = face_matcher_save_database(test_db_file, ids, tmpls, 2);
    ASSERT_TRUE(save_rc == 0, "Saved test binary embedding database");

    FaceConfig fcfg;
    memset(&fcfg, 0, sizeof(fcfg));
    fcfg.match_threshold = 0.25f;
    fcfg.match_margin = 0.05f;

    FaceMatcher *matcher = face_matcher_create(test_db_file, &fcfg);
    ASSERT_TRUE(matcher != NULL, "Loaded test FaceMatcher from file");
    ASSERT_TRUE(face_matcher_get_count(matcher) == 2, "Matcher loaded exactly 2 templates");

    /* Query exactly matching EMP-001 */
    float q1[FACE_EMBEDDING_DIM] = {0};
    q1[0] = 1.0f;
    MatchResult res1 = face_matcher_match(matcher, q1);
    ASSERT_TRUE(res1.is_matched && strcmp(res1.employee_id, "EMP-001") == 0, "Query matched EMP-001 with high score");
    ASSERT_TRUE(res1.score > 0.99f, "Score is ~1.0 for identical vector");

    /* Query orthogonal vector (no match) */
    float q_none[FACE_EMBEDDING_DIM] = {0};
    q_none[2] = 1.0f;
    MatchResult res_none = face_matcher_match(matcher, q_none);
    ASSERT_TRUE(!res_none.is_matched, "Orthogonal query vector correctly rejected");

    face_matcher_destroy(matcher);
    remove(test_db_file);
}

static void test_vision_tracking_and_voting(void) {
    printf("\n=== TEST 3: Centroid Tracking & Majority Voting ===\n");

    CentroidTracker tracker;
    centroid_tracker_init(&tracker, 100.0f);

    FaceBBox b1 = {100, 100, 200, 200};
    int64_t t1 = centroid_tracker_update_face(&tracker, &b1, 1.0);
    ASSERT_TRUE(t1 > 0, "Assigned new Track ID for initial detection");

    /* Nearby detection in next frame */
    FaceBBox b2 = {105, 102, 205, 202};
    int64_t t2 = centroid_tracker_update_face(&tracker, &b2, 1.1);
    ASSERT_TRUE(t1 == t2, "Matched same track ID for moved face bbox");

    /* Test majority voting queue */
    TrackVoteQueue vq = {0};
    track_vote_push(&vq, "EMP-A", 0.85f, 5);
    track_vote_push(&vq, "EMP-A", 0.90f, 5);

    char stable_id[128] = {0};
    float max_s = 0.0f;
    bool stable = track_vote_get_stable(&vq, 3, stable_id, sizeof(stable_id), &max_s);
    ASSERT_TRUE(!stable, "2 votes out of 3 required is not yet stable");

    track_vote_push(&vq, "EMP-A", 0.92f, 5);
    stable = track_vote_get_stable(&vq, 3, stable_id, sizeof(stable_id), &max_s);
    ASSERT_TRUE(stable && strcmp(stable_id, "EMP-A") == 0, "3 votes out of 3 required confirms EMP-A");
    ASSERT_TRUE(fabs(max_s - 0.92f) < 1e-4, "Peak score recorded properly");

    /* Test virtual line crossing direction */
    const char *dir_in = crossing_direction(-1, 1, true);
    ASSERT_TRUE(dir_in && strcmp(dir_in, "ENTER") == 0, "Crossing top to bottom is ENTER");

    const char *dir_out = crossing_direction(1, -1, true);
    ASSERT_TRUE(dir_out && strcmp(dir_out, "EXIT") == 0, "Crossing bottom to top is EXIT");
}

static void test_sqlite_database(void) {
    printf("\n=== TEST 4: SQLite Database Event Lifecycle ===\n");

    const char *test_db_file = "test_attendance.db";
    remove(test_db_file);

    AttendanceDB *db = attendance_db_open(test_db_file, 5);
    ASSERT_TRUE(db != NULL, "Opened test SQLite database");

    int64_t ev_id = attendance_db_create_pending_event(
        db,
        "test-cam",
        1001,
        "EMP-999",
        "ENTER",
        "PENDING_CONFIRMATION",
        0.88f
    );
    ASSERT_TRUE(ev_id > 0, "Created PENDING_CONFIRMATION attendance event");

    /* Check duplicate suppression */
    int64_t dup_id = attendance_db_create_pending_event(
        db,
        "test-cam",
        1001,
        "EMP-999",
        "ENTER",
        "PENDING_CONFIRMATION",
        0.89f
    );
    ASSERT_TRUE(dup_id == -1, "Duplicate event within cooldown properly suppressed");

    /* List events */
    AttendanceEvent events[10];
    int count = attendance_db_list_events(db, events, 10);
    ASSERT_TRUE(count == 1, "Listed 1 event from database");
    ASSERT_TRUE(strcmp(events[0].employee_id, "EMP-999") == 0, "Retrieved employee_id matches");
    ASSERT_TRUE(strcmp(events[0].status, "PENDING_CONFIRMATION") == 0, "Initial status is PENDING_CONFIRMATION");

    /* Respond with confirmation */
    int resp_rc = attendance_db_respond(db, ev_id, "confirm");
    ASSERT_TRUE(resp_rc == 0, "Responded to event with 'confirm'");

    count = attendance_db_list_events(db, events, 10);
    ASSERT_TRUE(strcmp(events[0].status, "CONFIRMED") == 0, "Status updated to CONFIRMED");
    ASSERT_TRUE(strcmp(events[0].event_type, "CHECK_IN") == 0, "Event type resolved to CHECK_IN");

    attendance_db_close(db);
    remove(test_db_file);
}

static void test_json(void) {
    printf("\n=== TEST 5: cJSON Serialization & Deserialization ===\n");

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "event_type", "CHECK_IN");
    cJSON_AddNumberToObject(root, "similarity", 0.954);
    cJSON_AddBoolToObject(root, "success", 1);

    char *json_str = cJSON_PrintUnformatted(root);
    ASSERT_TRUE(json_str != NULL, "Serialized cJSON object to string");

    cJSON *parsed = cJSON_Parse(json_str);
    ASSERT_TRUE(parsed != NULL, "Parsed JSON string back to cJSON object");
    cJSON *item = cJSON_GetObjectItem(parsed, "event_type");
    ASSERT_TRUE(item && strcmp(item->valuestring, "CHECK_IN") == 0, "Parsed field matches expected value");

    cJSON_Delete(parsed);
    cJSON_Delete(root);
    free(json_str);
}

int main(void) {
    printf("=================================================================\n");
    printf("     RUNNING ALL C UNIT & INTEGRATION TESTS                      \n");
    printf("=================================================================\n");

    test_config();
    test_matcher_and_normalization();
    test_vision_tracking_and_voting();
    test_sqlite_database();
    test_json();

    printf("\n=================================================================\n");
    printf(" TEST SUMMARY: Passed: %d | Failed: %d\n", tests_passed, tests_failed);
    printf("=================================================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
