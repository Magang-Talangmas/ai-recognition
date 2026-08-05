#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attendance/config.h"
#include "attendance/database.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <event_id> <action> [db_path]\n", argv[0]);
        printf("       %s --list [db_path]\n", argv[0]);
        printf("Actions: confirm | reject | break | temporary_exit | checkout | return_break | return_temporary\n");
        return 1;
    }

    AppConfig config;
    config_load("config.yaml", &config);

    const char *db_path = config.attendance.event_database;

    if (strcmp(argv[1], "--list") == 0) {
        if (argc > 2) db_path = argv[2];
        AttendanceDB *db = attendance_db_open(db_path, config.attendance.duplicate_cooldown_seconds);
        if (!db) {
            fprintf(stderr, "Failed to open database: %s\n", db_path);
            return 1;
        }

        AttendanceEvent events[50];
        int count = attendance_db_list_events(db, events, 50);

        printf("--------------------------------------------------------------------------------------------------------------------\n");
        printf("%-5s | %-12s | %-12s | %-6s | %-15s | %-6s | %-15s | %-20s\n",
               "ID", "CAMERA", "EMPLOYEE", "DIR", "EVENT_TYPE", "SIM", "STATUS", "DETECTED_AT");
        printf("--------------------------------------------------------------------------------------------------------------------\n");

        for (int i = 0; i < count; i++) {
            printf("%-5lld | %-12s | %-12s | %-6s | %-15s | %-6.2f | %-15s | %-20s\n",
                   (long long)events[i].id,
                   events[i].camera_id,
                   events[i].employee_id[0] ? events[i].employee_id : "UNKNOWN",
                   events[i].direction,
                   events[i].event_type,
                   events[i].similarity,
                   events[i].status,
                   events[i].detected_at);
        }
        printf("--------------------------------------------------------------------------------------------------------------------\n");
        printf("Total: %d events.\n", count);

        attendance_db_close(db);
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr, "Error: Missing action parameter.\n");
        return 1;
    }

    int64_t event_id = (int64_t)atoll(argv[1]);
    const char *action = argv[2];
    if (argc > 3) db_path = argv[3];

    AttendanceDB *db = attendance_db_open(db_path, config.attendance.duplicate_cooldown_seconds);
    if (!db) {
        fprintf(stderr, "Failed to open database: %s\n", db_path);
        return 1;
    }

    int rc = attendance_db_respond(db, event_id, action);
    if (rc == 0) {
        printf("[SUCCESS] Event %lld successfully updated with action '%s'.\n", (long long)event_id, action);
    } else {
        fprintf(stderr, "[FAILED] Could not update Event %lld (event may not exist or not in PENDING_CONFIRMATION status).\n", (long long)event_id);
    }

    attendance_db_close(db);
    return (rc == 0) ? 0 : 1;
}
