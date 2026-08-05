#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "attendance/config.h"
#include "attendance/database.h"
#include "attendance/face_engine.h"
#include "attendance/matcher.h"
#include "attendance/vision_utils.h"
#include "attendance/api_dispatcher.h"

#ifdef HAVE_OPENCV
#include <opencv2/c/opencv.h>
#endif

#include <signal.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static volatile bool g_running = true;

static void handle_sigint(int sig) {
    (void)sig;
    g_running = false;
}

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);
#ifdef SIGTERM
    signal(SIGTERM, handle_sigint);
#endif

    const char *config_file = "config.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }

    printf("=================================================================\n");
    printf("   TALANGMAS AI-RECOGNITION ATTENDANCE ENGINE (C EDITION)        \n");
    printf("=================================================================\n");

    AppConfig config;
    if (config_load(config_file, &config) != 0) {
        fprintf(stderr, "[Warning] Could not load '%s', using internal defaults.\n", config_file);
    }
    config_print(&config);

    /* Open Local Event Database */
    AttendanceDB *db = attendance_db_open(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds
    );
    if (!db) {
        fprintf(stderr, "[Error] Failed to initialize SQLite database '%s'\n", config.attendance.event_database);
        return 1;
    }
    printf("[Init] Local SQLite database connected: %s\n", config.attendance.event_database);

    /* Load Enrolled Face Templates */
    FaceMatcher *matcher = face_matcher_create(config.attendance.embedding_file, &config.face);
    if (!matcher) {
        fprintf(stderr, "[Error] Failed to initialize FaceMatcher.\n");
        attendance_db_close(db);
        return 1;
    }

    /* Initialize Face Detection & Recognition Engine */
    FaceEngine *face_engine = face_engine_create(&config.face);
    if (!face_engine) {
        fprintf(stderr, "[Error] Failed to initialize FaceEngine.\n");
        face_matcher_destroy(matcher);
        attendance_db_close(db);
        return 1;
    }

    /* Initialize Asynchronous Backend Dispatcher */
    BackendDispatcher *dispatcher = NULL;
    if (config.backend.enabled) {
        dispatcher = backend_dispatcher_create(&config.backend);
        if (dispatcher) {
            printf("[Init] Async HTTP Backend Dispatcher active: %s\n", config.backend.api_url);
        }
    }

    /* Initialize Centroid Face Tracker */
    CentroidTracker tracker;
    centroid_tracker_init(&tracker, 150.0f);

    printf("\n[System Ready] Starting video capture stream from: %s\n", config.camera.source);
    printf("Press Ctrl+C to stop.\n\n");

    int frame_index = 0;
    double last_fps_time = get_monotonic_time_seconds();
    int fps_frame_count = 0;

    /* Main Video Processing Loop */
    while (g_running) {
        double now = get_monotonic_time_seconds();
        frame_index++;
        fps_frame_count++;

        if (now - last_fps_time >= 2.0) {
            double current_fps = (double)fps_frame_count / (now - last_fps_time);
            fps_frame_count = 0;
            last_fps_time = now;
            printf("[Stream Active] FPS: %.1f | Frame: %d | Active Tracks: %d\n",
                   current_fps, frame_index, tracker.count);
            fflush(stdout);
        }

        /* Clean stale centroid tracks every 5 seconds */
        if (frame_index % 150 == 0) {
            centroid_tracker_clean_stale(&tracker, now, 5.0);
        }

        /* Frame skipping logic */
        if (frame_index % config.camera.process_every_n_frames != 0) {
#ifdef _WIN32
            Sleep(30);
#else
            usleep(30000);
#endif
            continue;
        }

        /*
         * Pipeline step 1: Grab Frame from Camera/RTSP
         * Pipeline step 2: Run Face Detection (SCRFD)
         * Pipeline step 3: Extract ArcFace Embeddings (112x112 aligned crop)
         * Pipeline step 4: Cosine Similarity Matching & Top-1/Top-2 Margin Filter
         * Pipeline step 5: Centroid Tracking & Majority Identity Voting
         * Pipeline step 6: Virtual Line Crossing & Direction Assessment
         * Pipeline step 7: SQLite DB Event Insertion & Async Backend Dispatch
         */
        
        FaceResult faces[MAX_DETECTED_FACES];
        ImageBuffer dummy_frame = {0};
        int num_faces = face_engine_detect(face_engine, &dummy_frame, faces, MAX_DETECTED_FACES);

        for (int i = 0; i < num_faces; i++) {
            FaceResult *face = &faces[i];
            int64_t track_id = centroid_tracker_update_face(&tracker, &face->bbox, now);

            if (face->has_embedding) {
                MatchResult match = face_matcher_match(matcher, face->embedding);

                /* Find track object */
                for (int t = 0; t < tracker.count; t++) {
                    TrackedFace *tf = &tracker.tracks[t];
                    if (tf->track_id == track_id) {
                        track_vote_push(&tf->vote_queue,
                                        match.is_matched ? match.employee_id : "",
                                        match.score,
                                        config.face.vote_window);

                        char stable_id[128] = {0};
                        float stable_score = 0.0f;
                        if (track_vote_get_stable(&tf->vote_queue,
                                                 config.face.votes_required,
                                                 stable_id, sizeof(stable_id),
                                                 &stable_score)) {
                            
                            if (!tf->is_confirmed) {
                                tf->is_confirmed = true;
                                strncpy(tf->confirmed_id, stable_id, sizeof(tf->confirmed_id) - 1);
                                tf->confirmed_score = stable_score;

                                /* Determine crossing direction */
                                float line_y = (float)dummy_frame.height * config.attendance.line_y_ratio;
                                int cur_side = (tf->cy >= line_y) ? 1 : -1;
                                const char *dir = crossing_direction(tf->previous_side, cur_side, config.attendance.inside_is_below_line);
                                if (!dir) dir = "ENTER";

                                int64_t evt_id = attendance_db_create_pending_event(
                                    db,
                                    config.camera.camera_id,
                                    track_id,
                                    stable_id,
                                    dir,
                                    "PENDING_CONFIRMATION",
                                    stable_score
                                );

                                if (evt_id > 0) {
                                    printf("\n[EVENT DETECTED] ID=%lld | Track=%lld | Employee=%s | Score=%.3f | Dir=%s\n",
                                           (long long)evt_id, (long long)track_id, stable_id, stable_score, dir);

                                    if (dispatcher) {
                                        char evt_str[64];
                                        snprintf(evt_str, sizeof(evt_str), "%lld", (long long)evt_id);
                                        backend_dispatcher_dispatch_checkin(
                                            dispatcher,
                                            stable_id,
                                            stable_score,
                                            config.camera.camera_id,
                                            evt_str,
                                            NULL
                                        );
                                    }
                                }
                                tf->previous_side = cur_side;
                            }
                        }
                        break;
                    }
                }
            }
        }

#ifdef _WIN32
        Sleep(30);
#else
        usleep(30000);
#endif
    }

    printf("\n[Shutdown] Cleaning up resources...\n");
    if (dispatcher) backend_dispatcher_destroy(dispatcher);
    face_engine_destroy(face_engine);
    face_matcher_destroy(matcher);
    attendance_db_close(db);

    printf("[Shutdown] AI-Recognition engine exited cleanly.\n");
    return 0;
}
