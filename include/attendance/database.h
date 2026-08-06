#ifndef ATTENDANCE_DATABASE_H
#define ATTENDANCE_DATABASE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t id;
    char event_id[64];  /* UUID v4 string */
    char camera_id[128];
    int64_t track_id;
    char employee_id[128];
    char original_candidate_id[128];
    char direction[32];
    char event_type[64];
    float similarity;
    char status[64];
    char employee_response[64];
    char detected_at[64];
    char responded_at[64];
} AttendanceEvent;

typedef struct AttendanceDB AttendanceDB;

/* Generate a compliant UUID v4 string (36 chars + null terminator) */
void attendance_generate_uuid_v4(char *out_uuid, size_t max_len);

/* Initialize SQLite database connection and create schema if not exists */
AttendanceDB *attendance_db_open(const char *db_path, int duplicate_cooldown_seconds);

/* Close database connection */
void attendance_db_close(AttendanceDB *db);

/* Create a new PENDING_CONFIRMATION event with generated UUID. Returns row id, or -1 if cooldown suppressed or error. */
int64_t attendance_db_create_pending_event(
    AttendanceDB *db,
    const char *camera_id,
    int64_t track_id,
    const char *employee_id,
    const char *direction,
    const char *event_type,
    float similarity,
    char *out_event_id,
    size_t event_id_len
);

/* Respond to an event (confirm, reject, break, temporary_exit, checkout, return_break, return_temporary) */
int attendance_db_respond(AttendanceDB *db, int64_t event_id, const char *action);

/* List recent attendance events. Returns number of events filled in events array. */
int attendance_db_list_events(AttendanceDB *db, AttendanceEvent *events, int max_events);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_DATABASE_H */
