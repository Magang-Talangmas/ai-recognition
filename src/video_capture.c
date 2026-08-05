#include "attendance/video_capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MAGIC_PACKET 0x53414D54 /* 'TMAS' */

struct VideoCapture {
    char source[512];
    int width;
    int height;
    uint8_t *frame_buffer;
    int frame_count;
    double start_time;
#ifdef _WIN32
    HANDLE h_read_pipe;
    HANDLE h_process;
#else
    FILE *pipe_fp;
#endif
    bool is_live;
};

static const char *find_python_executable(void) {
    static const char *candidates[] = {
        "D:\\kamera\\ai-recognition\\.venv\\Scripts\\python.exe",
        "..\\ai-recognition\\.venv\\Scripts\\python.exe",
        ".venv\\Scripts\\python.exe",
        "python.exe",
        "python3.exe",
        "python",
        NULL
    };

    for (int i = 0; candidates[i] != NULL; i++) {
#ifdef _WIN32
        if (GetFileAttributesA(candidates[i]) != INVALID_FILE_ATTRIBUTES) {
            return candidates[i];
        }
#else
        if (access(candidates[i], X_OK) == 0) {
            return candidates[i];
        }
#endif
    }
    return "python";
}

VideoCapture *video_capture_open(const char *source, int target_width, int target_height) {
    VideoCapture *cap = (VideoCapture *)calloc(1, sizeof(VideoCapture));
    if (!cap) return NULL;

    if (source && strlen(source) > 0) {
        strncpy(cap->source, source, sizeof(cap->source) - 1);
    } else {
        strncpy(cap->source, "0", sizeof(cap->source) - 1);
    }
    cap->width = (target_width > 0) ? target_width : 640;
    cap->height = (target_height > 0) ? target_height : 480;

    int buf_size = cap->width * cap->height * 3;
    cap->frame_buffer = (uint8_t *)calloc(1, buf_size);
    if (!cap->frame_buffer) {
        free(cap);
        return NULL;
    }

    const char *py = find_python_executable();
    const char *script_path = "scripts\\rtsp_feeder.py";
    if (GetFileAttributesA(script_path) == INVALID_FILE_ATTRIBUTES) {
        script_path = "camera-c\\scripts\\rtsp_feeder.py";
    }

    char cmdline[2048];
    snprintf(cmdline, sizeof(cmdline), "\"%s\" -u \"%s\" \"%s\" %d %d",
             py, script_path, cap->source, cap->width, cap->height);

    printf("[VideoCapture] Spawning real-time CCTV stream feeder...\n");
    printf("[VideoCapture] Stream Source: %s\n", cap->source);

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE h_read = NULL;
    HANDLE h_write = NULL;

    if (CreatePipe(&h_read, &h_write, &sa, 8 * 1024 * 1024)) {
        SetHandleInformation(h_read, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si;
        memset(&si, 0, sizeof(si));
        si.cb = sizeof(si);
        si.hStdOutput = h_write;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.dwFlags |= STARTF_USESTDHANDLES;

        PROCESS_INFORMATION pi;
        memset(&pi, 0, sizeof(pi));

        BOOL success = CreateProcessA(
            NULL,
            cmdline,
            NULL,
            NULL,
            TRUE,
            0,
            NULL,
            NULL,
            &si,
            &pi
        );

        CloseHandle(h_write);

        if (success) {
            cap->h_read_pipe = h_read;
            cap->h_process = pi.hProcess;
            CloseHandle(pi.hThread);
            cap->is_live = true;
            printf("[VideoCapture] CCTV stream feeder connected successfully.\n");
        } else {
            CloseHandle(h_read);
            fprintf(stderr, "[VideoCapture] Failed to launch feeder process (err=%lu)\n", GetLastError());
        }
    }
#else
    cap->pipe_fp = popen(cmdline, "r");
    if (cap->pipe_fp) {
        cap->is_live = true;
    }
#endif

    return cap;
}

static bool read_exact_bytes(VideoCapture *cap, void *buf, size_t bytes) {
    uint8_t *p = (uint8_t *)buf;
    size_t total = 0;

#ifdef _WIN32
    if (!cap->h_read_pipe) return false;
    while (total < bytes) {
        DWORD n = 0;
        DWORD to_read = (DWORD)(bytes - total);
        if (!ReadFile(cap->h_read_pipe, p + total, to_read, &n, NULL) || n == 0) {
            return false;
        }
        total += n;
    }
    return true;
#else
    if (!cap->pipe_fp) return false;
    while (total < bytes) {
        size_t n = fread(p + total, 1, bytes - total, cap->pipe_fp);
        if (n <= 0) return false;
        total += n;
    }
    return true;
#endif
}

bool video_capture_read_frame(
    VideoCapture *cap,
    ImageBuffer *out_frame,
    FaceResult *out_faces,
    int max_faces,
    int *num_faces_out,
    double now_time
) {
    (void)now_time;
    if (!cap || !cap->frame_buffer || !out_frame) return false;

    cap->frame_count++;
    int w = cap->width;
    int h = cap->height;
    uint8_t *buf = cap->frame_buffer;
    size_t img_bytes = (size_t)(w * h * 3);

    int detected_count = 0;
    bool got_live_frame = false;

    uint32_t header[4] = {0}; /* magic, width, height, num_faces */
    if (read_exact_bytes(cap, header, sizeof(header)) && header[0] == MAGIC_PACKET) {
        int pkt_w = (int)header[1];
        int pkt_h = (int)header[2];
        int pkt_faces = (int)header[3];

        if (pkt_w == w && pkt_h == h) {
            for (int i = 0; i < pkt_faces; i++) {
                struct {
                    float x1, y1, x2, y2;
                    float det_score;
                    float blur_score;
                    float lm_x[5];
                    float lm_y[5];
                } meta;

                float emb[FACE_EMBEDDING_DIM];

                if (read_exact_bytes(cap, &meta, sizeof(meta)) &&
                    read_exact_bytes(cap, emb, sizeof(emb))) {

                    if (out_faces && detected_count < max_faces) {
                        FaceResult *f = &out_faces[detected_count++];
                        memset(f, 0, sizeof(FaceResult));
                        f->bbox.x1 = meta.x1;
                        f->bbox.y1 = meta.y1;
                        f->bbox.x2 = meta.x2;
                        f->bbox.y2 = meta.y2;
                        f->detection_score = meta.det_score;
                        f->blur_score = meta.blur_score;
                        for (int k = 0; k < 5; k++) {
                            f->landmarks.x[k] = meta.lm_x[k];
                            f->landmarks.y[k] = meta.lm_y[k];
                        }
                        memcpy(f->embedding, emb, sizeof(emb));
                        f->has_embedding = true;
                        f->quality_ok = true;
                    }
                }
            }

            if (read_exact_bytes(cap, buf, img_bytes)) {
                got_live_frame = true;
            }
        }
    }

    if (!got_live_frame) {
        /* When connecting or between frames, keep clean neutral CCTV background */
        memset(buf, 20, img_bytes);
    }

    out_frame->data = buf;
    out_frame->width = w;
    out_frame->height = h;
    out_frame->channels = 3;
    out_frame->stride = w * 3;

    if (num_faces_out) {
        *num_faces_out = detected_count;
    }

    return true;
}

void video_capture_close(VideoCapture *cap) {
    if (!cap) return;
#ifdef _WIN32
    if (cap->h_read_pipe) {
        CloseHandle(cap->h_read_pipe);
        cap->h_read_pipe = NULL;
    }
    if (cap->h_process) {
        TerminateProcess(cap->h_process, 0);
        CloseHandle(cap->h_process);
        cap->h_process = NULL;
    }
#else
    if (cap->pipe_fp) {
        pclose(cap->pipe_fp);
        cap->pipe_fp = NULL;
    }
#endif
    if (cap->frame_buffer) {
        free(cap->frame_buffer);
    }
    free(cap);
}
