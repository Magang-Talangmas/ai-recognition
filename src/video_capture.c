#include "attendance/video_capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
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
    FILE *pipe_fp;
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
    char cmd[1024];

    const char *script_path = "scripts\\rtsp_feeder.py";
    if (GetFileAttributesA(script_path) == INVALID_FILE_ATTRIBUTES) {
        script_path = "camera-c\\scripts\\rtsp_feeder.py";
    }

    snprintf(cmd, sizeof(cmd), "\"%s\" -u \"%s\" \"%s\" %d %d",
             py, script_path, cap->source, cap->width, cap->height);

    printf("[VideoCapture] Launching real-time camera stream feeder...\n");
    printf("[VideoCapture] Executable: %s\n", py);

#ifdef _WIN32
    cap->pipe_fp = _popen(cmd, "rb");
#else
    cap->pipe_fp = popen(cmd, "r");
#endif

    if (cap->pipe_fp) {
        printf("[VideoCapture] Live video stream feeder process active.\n");
        cap->is_live = true;
    } else {
        printf("[VideoCapture] Warning: Could not launch live video feeder.\n");
    }

    return cap;
}

static bool read_exact(FILE *fp, void *buf, size_t bytes) {
    uint8_t *p = (uint8_t *)buf;
    size_t total = 0;
    while (total < bytes) {
        size_t n = fread(p + total, 1, bytes - total, fp);
        if (n <= 0) return false;
        total += n;
    }
    return true;
}

bool video_capture_read_frame(
    VideoCapture *cap,
    ImageBuffer *out_frame,
    FaceResult *out_faces,
    int max_faces,
    int *num_faces_out,
    double now_time
) {
    if (!cap || !cap->frame_buffer || !out_frame) return false;

    cap->frame_count++;
    int w = cap->width;
    int h = cap->height;
    uint8_t *buf = cap->frame_buffer;
    size_t img_bytes = (size_t)(w * h * 3);

    int detected_count = 0;
    bool got_live_frame = false;

    if (cap->pipe_fp) {
        uint32_t header[4] = {0}; /* magic, width, height, num_faces */
        if (read_exact(cap->pipe_fp, header, sizeof(header)) && header[0] == MAGIC_PACKET) {
            int pkt_w = (int)header[1];
            int pkt_h = (int)header[2];
            int pkt_faces = (int)header[3];

            if (pkt_w == w && pkt_h == h) {
                /* Read each detected face */
                for (int i = 0; i < pkt_faces; i++) {
                    struct {
                        float x1, y1, x2, y2;
                        float det_score;
                        float blur_score;
                        float lm_x[5];
                        float lm_y[5];
                    } meta;

                    float emb[FACE_EMBEDDING_DIM];

                    if (read_exact(cap->pipe_fp, &meta, sizeof(meta)) &&
                        read_exact(cap->pipe_fp, emb, sizeof(emb))) {

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

                /* Read raw image bytes */
                if (read_exact(cap->pipe_fp, buf, img_bytes)) {
                    got_live_frame = true;
                }
            }
        }
    }

    if (!got_live_frame) {
        /* Fallback synthetic test background */
        for (int y = 0; y < h; y++) {
            uint8_t bg_val = (uint8_t)(25 + (y * 20 / h));
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 3;
                buf[idx + 0] = bg_val;
                buf[idx + 1] = bg_val + 5;
                buf[idx + 2] = bg_val + 10;
            }
        }

        double t = now_time * 0.8;
        float person_y = (float)(h * 0.30 + (sin(t) * 0.5 + 0.5) * (h * 0.45));
        float person_x = (float)(w * 0.50 + cos(t * 0.7) * 60.0);

        int face_r = 35;
        int fcx = (int)person_x;
        int fcy = (int)person_y;

        for (int dy = -face_r; dy <= face_r; dy++) {
            for (int dx = -face_r; dx <= face_r; dx++) {
                if (dx * dx + dy * dy <= face_r * face_r) {
                    int px = fcx + dx;
                    int py = fcy + dy;
                    if (px >= 0 && px < w && py >= 0 && py < h) {
                        int idx = (py * w + px) * 3;
                        buf[idx + 0] = 195;
                        buf[idx + 1] = 160;
                        buf[idx + 2] = 140;
                    }
                }
            }
        }
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
    if (cap->pipe_fp) {
#ifdef _WIN32
        _pclose(cap->pipe_fp);
#else
        pclose(cap->pipe_fp);
#endif
        cap->pipe_fp = NULL;
    }
    if (cap->frame_buffer) {
        free(cap->frame_buffer);
    }
    free(cap);
}
