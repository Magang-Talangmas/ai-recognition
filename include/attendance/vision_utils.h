#ifndef ATTENDANCE_VISION_UTILS_H
#define ATTENDANCE_VISION_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VOTE_WINDOW 32
#define MAX_TRACKED_PEOPLE 128

typedef struct {
    char employee_id[128];
    bool has_id;
    float score;
} IdentityVote;

typedef struct {
    IdentityVote votes[MAX_VOTE_WINDOW];
    int count;
    int head;
    int capacity;
} TrackVoteQueue;

typedef struct {
    int64_t track_id;
    float cx;
    float cy;
    double last_seen_time;
    TrackVoteQueue vote_queue;
    char confirmed_id[128];
    float confirmed_score;
    bool is_confirmed;
    int previous_side; /* -1 = above, 1 = below, 0 = unset */
} TrackedFace;

typedef struct {
    TrackedFace tracks[MAX_TRACKED_PEOPLE];
    int count;
    float max_match_dist;
} CentroidTracker;

/* Initialize centroid tracker */
void centroid_tracker_init(CentroidTracker *tracker, float max_dist);

/* Match face bbox to existing track or assign new track ID */
int64_t centroid_tracker_update_face(
    CentroidTracker *tracker,
    const FaceBBox *bbox,
    double now_time
);

/* Clean tracks that have not been seen for longer than timeout_seconds */
void centroid_tracker_clean_stale(CentroidTracker *tracker, double now_time, double timeout_seconds);

/* Add identity vote to a track */
void track_vote_push(TrackVoteQueue *q, const char *employee_id, float score, int capacity);

/* Evaluate stable identity from vote queue based on required majority count */
bool track_vote_get_stable(
    const TrackVoteQueue *q,
    int required_votes,
    char *out_employee_id,
    size_t id_buf_size,
    float *out_max_score
);

/* Determine crossing direction across virtual line */
const char *crossing_direction(
    int previous_side,
    int current_side,
    bool inside_is_below_line
);

/* High-resolution monotonic time in seconds */
double get_monotonic_time_seconds(void);

/* Current UTC ISO8601 timestamp string (e.g. 2026-08-05T01:45:00Z) */
void get_iso8601_timestamp(char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_VISION_UTILS_H */
