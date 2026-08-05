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
#include "attendance/gui_preview.h"
#include "attendance/video_capture.h"

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

    /* Initialize Video Capture Stream */
    VideoCapture *cap = video_capture_open(config.camera.source, 640, 480);
    if (!cap) {
        fprintf(stderr, "[Error] Failed to initialize video capture stream.\n");
    }

    /* Initialize Native Windows GUI Preview Window */
    GuiWindow *win = NULL;
    if (config.camera.show_preview) {
        win = gui_window_create("Talangmas AI Recognition - Camera Stream (C Edition)", 640, 480);
        if (win) {
            printf("[GUI] Native Camera Stream Preview Window opened.\n");
        }
    }

    printf("\n[System Ready] Starting video capture stream from: %s\n", config.camera.source);
    printf("Press Ctrl+C or close preview window (Q / ESC) to stop.\n\n");

    int frame_index = 0;
    double last_fps_time = get_monotonic_time_seconds();
    int fps_frame_count = 0;
    double current_fps = 30.0;
    char last_event_msg[128] = {0};

    /* Main Video Processing Loop */
    while (g_running) {
        double now = get_monotonic_time_seconds();

        /* Process GUI window events (close button, Q, ESC) */
        if (win && !gui_window_process_events(win)) {
            printf("[GUI] Window close requested by user.\n");
            break;
        }

        frame_index++;
        fps_frame_count++;

        if (now - last_fps_time >= 2.0) {
            current_fps = (double)fps_frame_count / (now - last_fps_time);
            fps_frame_count = 0;
            last_fps_time = now;
            printf("[Stream Active] FPS: %.1f | Frame: %d\n", current_fps, frame_index);
            fflush(stdout);
        }

        /* Clean stale centroid tracks periodically */
        if (frame_index % 150 == 0) {
            centroid_tracker_clean_stale(&tracker, now, 5.0);
        }

        /* Grab Video Frame and Real Face Detections */
        ImageBuffer frame = {0};
        FaceResult faces[MAX_DETECTED_FACES];
        int num_faces = 0;
        double t_infer_start = get_monotonic_time_seconds();

        if (cap) {
            video_capture_read_frame(cap, &frame, faces, MAX_DETECTED_FACES, &num_faces, now);
        }

        /* Fallback face engine detection if needed */
        if (num_faces == 0 && face_engine) {
            num_faces = face_engine_detect(face_engine, &frame, faces, MAX_DETECTED_FACES);
        }
        double inference_ms = (get_monotonic_time_seconds() - t_infer_start) * 1000.0;

        for (int i = 0; i < num_faces; i++) {
            FaceResult *face = &faces[i];
            int64_t track_id = centroid_tracker_update_face(&tracker, &face->bbox, now);
            face->track_id = track_id;

            if (face->has_embedding) {
                MatchResult match = face_matcher_match(matcher, face->embedding);
                face->match_score = match.score;

                if (match.is_matched && match.employee_id[0] != '\0') {
                    strncpy(face->matched_name, match.employee_id, sizeof(face->matched_name) - 1);
                    face->is_recognized = true;
                }

                /* Find track object */
                for (int t = 0; t < tracker.count; t++) {
                    TrackedFace *tf = &tracker.tracks[t];
                    if (tf->track_id == track_id) {
                        track_vote_push(&tf->vote_queue,
                                        match.is_matched ? match.employee_id : "",
                                        match.score,
                                        config.face.vote_window);

                        char stable_name[128] = {0};
                        float stable_score = 0.0f;
                        if (track_vote_get_stable(&tf->vote_queue,
                                                 config.face.votes_required,
                                                 stable_name, sizeof(stable_name),
                                                 &stable_score)) {
                            
                            /* If confirmed through temporal voting, ensure face label has name */
                            strncpy(face->matched_name, stable_name, sizeof(face->matched_name) - 1);
                            face->match_score = stable_score;
                            face->is_recognized = true;

                            if (!tf->is_confirmed) {
                                tf->is_confirmed = true;
                                strncpy(tf->confirmed_id, stable_name, sizeof(tf->confirmed_id) - 1);
                                tf->confirmed_score = stable_score;

                                int64_t evt_id = attendance_db_create_pending_event(
                                    db,
                                    config.camera.camera_id,
                                    track_id,
                                    stable_name,
                                    "ABSENSI",
                                    "CONFIRMED",
                                    stable_score
                                );

                                if (evt_id > 0) {
                                    snprintf(last_event_msg, sizeof(last_event_msg),
                                             "[ABSENSI] %s", stable_name);

                                    printf("\n[ABSENSI] %s terdeteksi\n", stable_name);

                                    if (dispatcher) {
                                        char evt_str[64];
                                        snprintf(evt_str, sizeof(evt_str), "%lld", (long long)evt_id);
                                        backend_dispatcher_dispatch_checkin(
                                            dispatcher,
                                            stable_name,
                                            stable_score,
                                            config.camera.camera_id,
                                            evt_str,
                                            NULL
                                        );
                                    }
                                }
                            }
                        } else if (tf->is_confirmed && tf->confirmed_id[0] != '\0') {
                            strncpy(face->matched_name, tf->confirmed_id, sizeof(face->matched_name) - 1);
                            face->match_score = tf->confirmed_score;
                            face->is_recognized = true;
                        }
                        break;
                    }
                }
            }
        }

        /* Frame-level identity de-duplication:
           Ensure that no identity/person is displayed double in the same frame.
           If multiple detected faces claim the same name, ONLY the face with the highest percentage/score wins!
           The lower-scoring face(s) become Unknown. */
        for (int i = 0; i < num_faces; i++) {
            if (!faces[i].is_recognized || faces[i].matched_name[0] == '\0') continue;
            for (int j = i + 1; j < num_faces; j++) {
                if (!faces[j].is_recognized || faces[j].matched_name[0] == '\0') continue;
                if (strcmp(faces[i].matched_name, faces[j].matched_name) == 0) {
                    if (faces[i].match_score >= faces[j].match_score) {
                        faces[j].is_recognized = false;
                        faces[j].matched_name[0] = '\0';
                    } else {
                        faces[i].is_recognized = false;
                        faces[i].matched_name[0] = '\0';
                        break;
                    }
                }
            }
        }

        /* Render to native Windows GUI Window */
        if (win) {
            gui_window_render(
                win,
                &frame,
                faces,
                num_faces,
                (float)current_fps,
                (float)inference_ms,
                last_event_msg
            );
        }

#ifdef _WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }

    printf("\n[Shutdown] Cleaning up resources...\n");
    if (win) gui_window_destroy(win);
    if (cap) video_capture_close(cap);
    if (dispatcher) backend_dispatcher_destroy(dispatcher);
    face_engine_destroy(face_engine);
    face_matcher_destroy(matcher);
    attendance_db_close(db);

    printf("[Shutdown] AI-Recognition engine exited cleanly.\n");
    return 0;
}
