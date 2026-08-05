#include "attendance/vision_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

double get_monotonic_time_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

void get_iso8601_timestamp(char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) return;
    time_t now = time(NULL);
    struct tm tm_utc;
#ifdef _WIN32
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

void centroid_tracker_init(CentroidTracker *tracker, float max_dist) {
    if (!tracker) return;
    memset(tracker, 0, sizeof(CentroidTracker));
    tracker->max_match_dist = (max_dist > 0.0f) ? max_dist : 120.0f;
}

int64_t centroid_tracker_update_face(
    CentroidTracker *tracker,
    const FaceBBox *bbox,
    double now_time
) {
    if (!tracker || !bbox) return 0;

    float cx = (bbox->x1 + bbox->x2) / 2.0f;
    float cy = (bbox->y1 + bbox->y2) / 2.0f;

    int best_track_idx = -1;
    float min_dist = 1e9f;

    for (int i = 0; i < tracker->count; i++) {
        TrackedFace *tf = &tracker->tracks[i];
        float dx = cx - tf->cx;
        float dy = cy - tf->cy;
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < min_dist && dist <= tracker->max_match_dist) {
            min_dist = dist;
            best_track_idx = i;
        }
    }

    if (best_track_idx >= 0) {
        TrackedFace *tf = &tracker->tracks[best_track_idx];
        tf->cx = cx;
        tf->cy = cy;
        tf->last_seen_time = now_time;
        return tf->track_id;
    }

    /* Assign new track */
    if (tracker->count < MAX_TRACKED_PEOPLE) {
        int idx = tracker->count++;
        TrackedFace *tf = &tracker->tracks[idx];
        memset(tf, 0, sizeof(TrackedFace));
        
        static int64_t next_track_id = 1000;
        tf->track_id = next_track_id++;
        tf->cx = cx;
        tf->cy = cy;
        tf->last_seen_time = now_time;
        return tf->track_id;
    }

    return 0;
}

void centroid_tracker_clean_stale(CentroidTracker *tracker, double now_time, double timeout_seconds) {
    if (!tracker) return;

    int active_count = 0;
    for (int i = 0; i < tracker->count; i++) {
        if (now_time - tracker->tracks[i].last_seen_time <= timeout_seconds) {
            if (active_count != i) {
                tracker->tracks[active_count] = tracker->tracks[i];
            }
            active_count++;
        }
    }
    tracker->count = active_count;
}

void track_vote_push(TrackVoteQueue *q, const char *employee_id, float score, int capacity) {
    if (!q) return;
    if (capacity <= 0 || capacity > MAX_VOTE_WINDOW) capacity = MAX_VOTE_WINDOW;
    q->capacity = capacity;

    int idx = (q->head + q->count) % capacity;
    if (q->count == capacity) {
        /* Overwrite oldest element */
        q->head = (q->head + 1) % capacity;
    } else {
        q->count++;
    }

    IdentityVote *vote = &q->votes[idx];
    memset(vote, 0, sizeof(IdentityVote));
    if (employee_id && strlen(employee_id) > 0) {
        strncpy(vote->employee_id, employee_id, sizeof(vote->employee_id) - 1);
        vote->has_id = true;
    }
    vote->score = score;
}

bool track_vote_get_stable(
    const TrackVoteQueue *q,
    int required_votes,
    char *out_employee_id,
    size_t id_buf_size,
    float *out_max_score
) {
    if (!q || q->count == 0 || required_votes <= 0) return false;

    typedef struct {
        char id[128];
        int count;
        float max_score;
    } VoteCount;

    VoteCount candidates[MAX_VOTE_WINDOW];
    int cand_num = 0;

    for (int i = 0; i < q->count; i++) {
        int idx = (q->head + i) % q->capacity;
        const IdentityVote *vote = &q->votes[idx];
        if (!vote->has_id) continue;

        int found = -1;
        for (int k = 0; k < cand_num; k++) {
            if (strcmp(candidates[k].id, vote->employee_id) == 0) {
                found = k;
                break;
            }
        }

        if (found >= 0) {
            candidates[found].count++;
            if (vote->score > candidates[found].max_score) {
                candidates[found].max_score = vote->score;
            }
        } else if (cand_num < MAX_VOTE_WINDOW) {
            strncpy(candidates[cand_num].id, vote->employee_id, sizeof(candidates[cand_num].id) - 1);
            candidates[cand_num].count = 1;
            candidates[cand_num].max_score = vote->score;
            cand_num++;
        }
    }

    int best_cand = -1;
    int max_votes = 0;
    for (int k = 0; k < cand_num; k++) {
        if (candidates[k].count > max_votes) {
            max_votes = candidates[k].count;
            best_cand = k;
        }
    }

    if (best_cand >= 0 && max_votes >= required_votes) {
        if (out_employee_id && id_buf_size > 0) {
            strncpy(out_employee_id, candidates[best_cand].id, id_buf_size - 1);
            out_employee_id[id_buf_size - 1] = '\0';
        }
        if (out_max_score) {
            *out_max_score = candidates[best_cand].max_score;
        }
        return true;
    }

    return false;
}

const char *crossing_direction(
    int previous_side,
    int current_side,
    bool inside_is_below_line
) {
    if (previous_side == 0 || previous_side == current_side) {
        return NULL;
    }

    bool entered_inside = inside_is_below_line ? (current_side == 1) : (current_side == -1);
    return entered_inside ? "ENTER" : "EXIT";
}
