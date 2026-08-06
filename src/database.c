#include "attendance/database.h"
#include "attendance/vision_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__has_include)
  #if __has_include(<sqlite3.h>)
    #include <sqlite3.h>
  #else
    #include "third_party/sqlite3.h"
  #endif
#else
  #include "third_party/sqlite3.h"
#endif

#ifdef _WIN32
#include <direct.h>
#include <objbase.h>
#define mkdir_compat(p) _mkdir(p)
#else
#include <sys/stat.h>
#define mkdir_compat(p) mkdir(p, 0755)
#endif

struct AttendanceDB {
    sqlite3 *db;
    int duplicate_cooldown_seconds;
};

static const char *SCHEMA_SQL =
    "CREATE TABLE IF NOT EXISTS attendance_events (\n"
    "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "    event_id TEXT,\n"
    "    camera_id TEXT NOT NULL,\n"
    "    track_id INTEGER NOT NULL,\n"
    "    employee_id TEXT,\n"
    "    original_candidate_id TEXT,\n"
    "    direction TEXT NOT NULL,\n"
    "    event_type TEXT NOT NULL,\n"
    "    similarity REAL NOT NULL,\n"
    "    status TEXT NOT NULL,\n"
    "    employee_response TEXT,\n"
    "    detected_at TEXT NOT NULL,\n"
    "    responded_at TEXT\n"
    ");\n";

void attendance_generate_uuid_v4(char *out_uuid, size_t max_len) {
    if (!out_uuid || max_len < 37) return;

#ifdef _WIN32
    GUID guid;
    if (SUCCEEDED(CoCreateGuid(&guid))) {
        snprintf(out_uuid, max_len,
                 "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 (unsigned long)guid.Data1,
                 (unsigned int)guid.Data2,
                 (unsigned int)guid.Data3,
                 guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],
                 guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);
        return;
    }
#endif

    /* Portable fallback random UUID v4 */
    uint8_t bytes[16];
    for (int i = 0; i < 16; i++) {
        bytes[i] = (uint8_t)(rand() & 0xFF);
    }
    bytes[6] = (bytes[6] & 0x0F) | 0x40; /* Version 4 */
    bytes[8] = (bytes[8] & 0x3F) | 0x80; /* Variant 1 */
    snprintf(out_uuid, max_len,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5],
             bytes[6], bytes[7],
             bytes[8], bytes[9],
             bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

static void ensure_parent_dir(const char *filepath) {
    char temp[512];
    strncpy(temp, filepath, sizeof(temp) - 1);
    temp[sizeof(temp) - 1] = '\0';

    char *p = temp;
    while (*p) {
        if (*p == '/' || *p == '\\') {
            char orig = *p;
            *p = '\0';
            if (strlen(temp) > 0) {
                mkdir_compat(temp);
            }
            *p = orig;
        }
        p++;
    }
}

AttendanceDB *attendance_db_open(const char *db_path, int duplicate_cooldown_seconds) {
    ensure_parent_dir(db_path);

    AttendanceDB *db_obj = (AttendanceDB *)calloc(1, sizeof(AttendanceDB));
    if (!db_obj) return NULL;

    db_obj->duplicate_cooldown_seconds = duplicate_cooldown_seconds;

    int rc = sqlite3_open_v2(
        db_path,
        &db_obj->db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        NULL
    );

    if (rc != SQLITE_OK) {
        fprintf(stderr, "[DB ERROR] Cannot open SQLite database %s: %s\n",
                db_path, sqlite3_errmsg(db_obj->db));
        if (db_obj->db) sqlite3_close(db_obj->db);
        free(db_obj);
        return NULL;
    }

    char *errmsg = NULL;
    rc = sqlite3_exec(db_obj->db, SCHEMA_SQL, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "[DB ERROR] Schema migration failed: %s\n", errmsg ? errmsg : "unknown");
        sqlite3_free(errmsg);
        sqlite3_close(db_obj->db);
        free(db_obj);
        return NULL;
    }

    /* Auto-migrate event_id column if table already exists without it */
    sqlite3_exec(db_obj->db, "ALTER TABLE attendance_events ADD COLUMN event_id TEXT;", NULL, NULL, NULL);

    /* Enable WAL mode for high concurrency */
    sqlite3_exec(db_obj->db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);

    return db_obj;
}

void attendance_db_close(AttendanceDB *db) {
    if (!db) return;
    if (db->db) {
        sqlite3_close(db->db);
        db->db = NULL;
    }
    free(db);
}

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
) {
    if (!db || !db->db || !camera_id || !employee_id || !direction || !event_type) {
        return -1;
    }

    /* Check duplicate cooldown within window */
    const char *CHECK_COOLDOWN_SQL =
        "SELECT id, detected_at FROM attendance_events "
        "WHERE camera_id = ? AND employee_id = ? AND event_type = ? "
        "  AND status = 'PENDING_CONFIRMATION' "
        "ORDER BY id DESC LIMIT 1;";

    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db->db, CHECK_COOLDOWN_SQL, -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, camera_id, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, employee_id, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, event_type, -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            /* Existing pending event exists, suppress duplicate */
            sqlite3_finalize(stmt);
            return -1;
        }
        sqlite3_finalize(stmt);
    }

    char now_iso[64];
    get_iso8601_timestamp(now_iso, sizeof(now_iso));

    char uuid_v4[64];
    attendance_generate_uuid_v4(uuid_v4, sizeof(uuid_v4));
    if (out_event_id && event_id_len > 0) {
        strncpy(out_event_id, uuid_v4, event_id_len - 1);
        out_event_id[event_id_len - 1] = '\0';
    }

    const char *INSERT_SQL =
        "INSERT INTO attendance_events ("
        "    event_id, camera_id, track_id, employee_id, original_candidate_id, "
        "    direction, event_type, similarity, status, detected_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_CONFIRMATION', ?);";

    rc = sqlite3_prepare_v2(db->db, INSERT_SQL, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "[DB ERROR] Failed to prepare insert statement: %s\n", sqlite3_errmsg(db->db));
        return -1;
    }

    sqlite3_bind_text(stmt, 1, uuid_v4, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, camera_id, -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 3, track_id);
    sqlite3_bind_text(stmt, 4, employee_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, employee_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, direction, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, event_type, -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 8, (double)similarity);
    sqlite3_bind_text(stmt, 9, now_iso, -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        fprintf(stderr, "[DB ERROR] Insert event failed: %s\n", sqlite3_errmsg(db->db));
        return -1;
    }

    return (int64_t)sqlite3_last_insert_rowid(db->db);
}

int attendance_db_respond(AttendanceDB *db, int64_t event_id, const char *action) {
    if (!db || !db->db || !action) return -1;

    char now_iso[64];
    get_iso8601_timestamp(now_iso, sizeof(now_iso));

    if (strcmp(action, "reject") == 0) {
        const char *REJECT_SQL =
            "UPDATE attendance_events SET "
            "    employee_id = NULL, "
            "    status = 'UNRESOLVED_RECOGNITION', "
            "    employee_response = 'NOT_ME', "
            "    responded_at = ? "
            "WHERE id = ? AND status = 'PENDING_CONFIRMATION';";

        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db->db, REJECT_SQL, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, now_iso, -1, SQLITE_STATIC);
            sqlite3_bind_int64(stmt, 2, event_id);
            int step_rc = sqlite3_step(stmt);
            sqlite3_finalize(stmt);
            return (step_rc == SQLITE_DONE) ? 0 : -1;
        }
        return -1;
    }

    const char *resolved_event = NULL;
    if (strcmp(action, "confirm") == 0) resolved_event = "CHECK_IN";
    else if (strcmp(action, "break") == 0) resolved_event = "START_BREAK";
    else if (strcmp(action, "temporary_exit") == 0) resolved_event = "TEMPORARY_EXIT";
    else if (strcmp(action, "checkout") == 0) resolved_event = "CHECK_OUT";
    else if (strcmp(action, "return_break") == 0) resolved_event = "RETURN_FROM_BREAK";
    else if (strcmp(action, "return_temporary") == 0) resolved_event = "RETURN_FROM_TEMPORARY_EXIT";
    else {
        fprintf(stderr, "[DB ERROR] Unsupported action: %s\n", action);
        return -1;
    }

    char upper_action[64];
    size_t act_len = strlen(action);
    for (size_t i = 0; i < act_len && i < sizeof(upper_action) - 1; i++) {
        char c = action[i];
        upper_action[i] = (c >= 'a' && c <= 'z') ? (c - 32) : c;
    }
    upper_action[act_len] = '\0';

    const char *CONFIRM_SQL =
        "UPDATE attendance_events SET "
        "    event_type = ?, "
        "    status = 'CONFIRMED', "
        "    employee_response = ?, "
        "    responded_at = ? "
        "WHERE id = ? AND status = 'PENDING_CONFIRMATION';";

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db->db, CONFIRM_SQL, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, resolved_event, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, upper_action, -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, now_iso, -1, SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 4, event_id);
        int step_rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        return (step_rc == SQLITE_DONE) ? 0 : -1;
    }

    return -1;
}

int attendance_db_list_events(AttendanceDB *db, AttendanceEvent *events, int max_events) {
    if (!db || !db->db || !events || max_events <= 0) return 0;

    const char *LIST_SQL =
        "SELECT id, IFNULL(event_id, ''), camera_id, track_id, IFNULL(employee_id, ''), "
        "       direction, event_type, similarity, status, "
        "       IFNULL(employee_response, ''), detected_at, IFNULL(responded_at, '') "
        "FROM attendance_events ORDER BY id DESC LIMIT ?;";

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db->db, LIST_SQL, -1, &stmt, NULL) != SQLITE_OK) {
        return 0;
    }

    sqlite3_bind_int(stmt, 1, max_events);

    int count = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && count < max_events) {
        AttendanceEvent *ev = &events[count];
        memset(ev, 0, sizeof(AttendanceEvent));

        ev->id = sqlite3_column_int64(stmt, 0);
        strncpy(ev->event_id, (const char *)sqlite3_column_text(stmt, 1), sizeof(ev->event_id) - 1);
        strncpy(ev->camera_id, (const char *)sqlite3_column_text(stmt, 2), sizeof(ev->camera_id) - 1);
        ev->track_id = sqlite3_column_int64(stmt, 3);
        strncpy(ev->employee_id, (const char *)sqlite3_column_text(stmt, 4), sizeof(ev->employee_id) - 1);
        strncpy(ev->direction, (const char *)sqlite3_column_text(stmt, 5), sizeof(ev->direction) - 1);
        strncpy(ev->event_type, (const char *)sqlite3_column_text(stmt, 6), sizeof(ev->event_type) - 1);
        ev->similarity = (float)sqlite3_column_double(stmt, 7);
        strncpy(ev->status, (const char *)sqlite3_column_text(stmt, 8), sizeof(ev->status) - 1);
        strncpy(ev->employee_response, (const char *)sqlite3_column_text(stmt, 9), sizeof(ev->employee_response) - 1);
        strncpy(ev->detected_at, (const char *)sqlite3_column_text(stmt, 10), sizeof(ev->detected_at) - 1);
        strncpy(ev->responded_at, (const char *)sqlite3_column_text(stmt, 11), sizeof(ev->responded_at) - 1);

        count++;
    }

    sqlite3_finalize(stmt);
    return count;
}
